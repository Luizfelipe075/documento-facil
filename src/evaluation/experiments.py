"""
src/evaluation/experiments.py

TASK 17 — Experimentos de avaliação do Documento Fácil.

O que este módulo faz
---------------------
1. Carrega o corpus (documentos já coletados) e o conjunto de perguntas.
2. Para cada configuração de chunking (tamanho x sobreposição):
     chunking -> embeddings -> índice ChromaDB ISOLADO -> avaliação de recuperação.
3. Compara as configurações (Recall@K, MRR, latência, nº de chunks).
4. Opcional: avalia a GERAÇÃO (LLM) na melhor configuração, com checagens
   heurísticas automáticas + planilha para avaliação manual.
5. Grava resultados em data/experiments/<timestamp>/ (JSON, CSV e relatório .md).

Dois níveis de medição (ver metrics.py)
---------------------------------------
- Por DOCUMENTO: o documento esperado aparece no Top-K? (equivale ao
  evaluate_retrieval existente). Trivial quando só há 1 documento indexado.
- Por EVIDÊNCIA: algum chunk do Top-K contém o trecho que responde à pergunta?
  Este é o nível que realmente diferencia as configurações.

Os índices dos experimentos ficam em diretório temporário e NÃO tocam o
./vectorstore usado pelo app.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.evaluation import metrics as M

logger = logging.getLogger(__name__)

# Perguntas usam o ID simbólico "poc_ldb_l9394" (gerado por scripts/poc.py).
# O coletor gera IDs diferentes (slug + hash da URL). Os aliases fazem a ponte.
DEFAULT_ALIASES: dict[str, list[str]] = {"poc_ldb_l9394": ["l9394"]}

# scripts/poc.py --demo salva o HTML como poc_ldb_demo.html mas indexa como poc_ldb_l9394.
FILE_STEM_TO_DOC_ID: dict[str, str] = {"poc_ldb_demo": "poc_ldb_l9394"}

DEFAULT_CHUNK_SIZES = [250, 500, 800]
DEFAULT_OVERLAPS = [0, 100]
DEFAULT_K_VALUES = [1, 3, 5]


# ======================================================================
# Configuração
# ======================================================================

class ExperimentConfig:
    """Uma combinação de parâmetros de chunking a ser avaliada."""

    def __init__(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) deve ser menor que chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"cs{self.chunk_size}_ov{self.overlap}"

    def __repr__(self) -> str:
        return f"ExperimentConfig({self.name})"


def make_grid(
    chunk_sizes: list[int] = None,
    overlaps: list[int] = None,
) -> list[ExperimentConfig]:
    """Produto cartesiano tamanho x sobreposição, ignorando combinações inválidas."""
    chunk_sizes = chunk_sizes or DEFAULT_CHUNK_SIZES
    overlaps = overlaps or DEFAULT_OVERLAPS
    grid = []
    for size in chunk_sizes:
        for ov in overlaps:
            if ov < size:
                grid.append(ExperimentConfig(size, ov))
            else:
                logger.warning("Combinação ignorada: chunk_size=%d overlap=%d", size, ov)
    return grid


# ======================================================================
# Carregamento de dados
# ======================================================================

def load_questions(path: Path) -> list[dict]:
    """Carrega test_questions.json e garante um 'id' em cada pergunta."""
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    for i, q in enumerate(questions, start=1):
        q.setdefault("id", f"q{i:02d}")
        q.setdefault("category", "sem_categoria")
        q.setdefault("expected_documents", [])
    return questions


def load_evidence(path: Optional[Path], questions: list[dict]) -> dict[str, list[str]]:
    """
    Evidências por pergunta: textos que, se presentes num chunk, indicam que o
    chunk responde à pergunta. Cada evidência é uma lista de ALTERNATIVAS
    (basta uma casar). Fontes, em ordem de prioridade:
      1. campo "evidence" dentro da própria pergunta;
      2. arquivo sidecar {"q01": ["texto a", "texto b"], ...}.
    """
    sidecar: dict[str, list[str]] = {}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        sidecar = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, list)}

    evidence = {}
    for q in questions:
        ev = q.get("evidence") or sidecar.get(q["id"])
        if ev:
            evidence[q["id"]] = list(ev)
    return evidence


def load_aliases(path: Optional[Path]) -> dict[str, list[str]]:
    """Aliases de document_id: padrão do projeto + arquivo opcional."""
    aliases = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for key, values in raw.items():
            if key.startswith("_") or not isinstance(values, list):
                continue
            aliases[key] = sorted(set(aliases.get(key, []) + values))
    return aliases


def _extract_one(file_path: Path, content_type: str, doc_id: str, meta: dict) -> Optional[tuple[str, dict]]:
    """Extrai o texto de um arquivo e devolve (texto_bruto, metadados_atualizados)."""
    meta = dict(meta)
    if content_type == "pdf":
        from src.extraction.pdf import extract_pdf
        result = extract_pdf(file_path, doc_id)
        if result is None or not result.full_text.strip():
            logger.warning("PDF sem texto extraível: %s", doc_id)
            return None
        meta.setdefault("title", file_path.stem)
        return result.full_text, meta

    from src.extraction.html import extract_html
    result = extract_html(file_path, doc_id, meta.get("source_url") or file_path.resolve().as_uri())
    if result is None or not result.body.strip():
        logger.warning("HTML sem conteúdo: %s", doc_id)
        return None
    meta["title"] = meta.get("title") or result.title or file_path.stem
    meta["publication_date"] = meta.get("publication_date") or result.date
    meta["organization"] = meta.get("organization") or result.organization
    return result.body, meta


def load_corpus(
    root: Path,
    documents_file: Optional[Path] = None,
    corpus_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Carrega e limpa os documentos do corpus.

    Fontes (em ordem):
      - corpus_dir: se informado, usa SOMENTE os .html/.pdf desse diretório;
      - senão: data/processed/documents.jsonl (pipeline completo) e, além disso,
        qualquer arquivo em data/raw/html|pdf ainda não listado (ex.: o HTML da PoC).

    Retorna lista de {"document_id", "meta", "text"} com texto já limpo.
    """
    from src.processing.cleaner import clean_text

    root = Path(root)
    corpus: list[dict] = []
    seen_files: set[Path] = set()

    def add(doc_id: str, file_path: Path, content_type: str, meta: dict) -> None:
        extracted = _extract_one(file_path, content_type, doc_id, meta)
        seen_files.add(file_path.resolve())
        if extracted is None:
            return
        raw_text, full_meta = extracted
        text = clean_text(raw_text, doc_id)
        if not text.strip():
            logger.warning("Texto vazio após limpeza: %s", doc_id)
            return
        full_meta.setdefault("source_url", file_path.resolve().as_uri())
        full_meta.setdefault("organization", "Documento local")
        corpus.append({"document_id": doc_id, "meta": full_meta, "text": text})

    if corpus_dir is None:
        docs_file = Path(documents_file) if documents_file else root / "data" / "processed" / "documents.jsonl"
        if docs_file.exists():
            with open(docs_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    doc = json.loads(line)
                    fp = root / doc.get("file_path", "")
                    if not doc.get("file_path") or not fp.exists():
                        logger.warning("Arquivo não encontrado para %s: %s", doc.get("document_id"), fp)
                        continue
                    add(doc["document_id"], fp, doc.get("content_type", "html"), doc)
        scan_dirs = [root / "data" / "raw" / "html", root / "data" / "raw" / "pdf"]
    else:
        scan_dirs = [Path(corpus_dir)]

    existing_ids = {c["document_id"] for c in corpus}
    for d in scan_dirs:
        if not d.exists():
            continue
        for fp in sorted(d.iterdir()):
            suffix = fp.suffix.lower()
            if suffix not in (".html", ".htm", ".pdf") or fp.resolve() in seen_files:
                continue
            doc_id = FILE_STEM_TO_DOC_ID.get(fp.stem, fp.stem)
            if doc_id in existing_ids:
                continue
            add(doc_id, fp, "pdf" if suffix == ".pdf" else "html", {})
            existing_ids.add(doc_id)

    logger.info("Corpus carregado: %d documento(s): %s", len(corpus), [c["document_id"] for c in corpus])
    return corpus


def build_chunks(corpus: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Aplica o chunker do projeto a todos os documentos do corpus."""
    from src.processing.chunker import chunk_document

    chunks: list[dict] = []
    for doc in corpus:
        chunks.extend(chunk_document(
            text=doc["text"],
            document_id=doc["document_id"],
            metadata=doc["meta"],
            chunk_size=chunk_size,
            overlap=overlap,
        ))
    return chunks


def check_evidence(corpus: list[dict], questions: list[dict], evidence: dict[str, list[str]]) -> list[dict]:
    """
    Valida as evidências contra o TEXTO COMPLETO do corpus (antes do chunking).
    Evidência que não aparece em nenhum lugar é incorreta/desatualizada e
    distorceria as métricas — conserte-a em data/questions/evidence_ldb.json.
    """
    full_text = M.normalize(" ".join(d["text"] for d in corpus))
    report = []
    for q in questions:
        ev = evidence.get(q["id"])
        if not ev:
            continue
        found = [e for e in ev if M.normalize(e) in full_text]
        report.append({"id": q["id"], "question": q["question"], "alternatives": ev,
                       "found": found, "ok": bool(found)})
    return report


# ======================================================================
# Índice e utilitários
# ======================================================================

def _close_index(index) -> None:
    """Libera locks de arquivo do ChromaDB (importante no Windows)."""
    try:
        client = getattr(index, "_client", None)
        if client is not None and hasattr(client, "_system"):
            client._system.stop()
    except Exception:  # noqa: BLE001 - limpeza best-effort
        pass
    index._client = None
    index._collection = None


class QueryCache:
    """Embeddings de perguntas são independentes da configuração: calcula 1 vez."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: dict[str, object] = {}

    def get(self, query: str):
        if query not in self._cache:
            self._cache[query] = self.embedder.embed_query(query)
        return self._cache[query]


def _tokenizer_name() -> str:
    from src.processing.chunker import Chunker
    enc = Chunker(chunk_size=10, overlap=0).enc
    return "aproximado por palavras (fallback offline)" if enc.__class__.__name__ == "_RegexEncoding" \
        else "tiktoken cl100k_base"


# ======================================================================
# Experimento de RECUPERAÇÃO
# ======================================================================

def run_retrieval_experiment(
    config: ExperimentConfig,
    corpus: list[dict],
    questions: list[dict],
    embedder,
    query_cache: QueryCache,
    workdir: Path,
    evidence: Optional[dict[str, list[str]]] = None,
    aliases: Optional[dict[str, list[str]]] = None,
    k_values: Optional[list[int]] = None,
) -> tuple[dict, object]:
    """
    Executa UMA configuração. Retorna (resultado, índice_aberto).
    O chamador decide quando fechar o índice (útil para reutilizá-lo na geração).
    """
    from src.retrieval.index import VectorIndex

    evidence = evidence or {}
    aliases = aliases if aliases is not None else DEFAULT_ALIASES
    k_values = sorted(k_values or DEFAULT_K_VALUES)
    max_k = max(k_values)

    logger.info("=== Configuração %s ===", config.name)

    chunks = build_chunks(corpus, config.chunk_size, config.overlap)
    if not chunks:
        raise RuntimeError(f"Nenhum chunk gerado para {config.name}. O corpus está vazio?")
    norm_chunks = [M.normalize(c["text"]) for c in chunks]
    chunk_tokens = {c["chunk_id"]: c["token_count"] for c in chunks}

    t0 = time.perf_counter()
    index = VectorIndex(persist_dir=workdir / config.name, embedder=embedder)
    index.build(chunks, embedder=embedder)
    build_seconds = time.perf_counter() - t0

    rows: list[dict] = []
    search_ms: list[float] = []

    for q in questions:
        expected = q.get("expected_documents", [])
        answerable = bool(expected)
        qvec = query_cache.get(q["question"])

        t1 = time.perf_counter()
        hits = index.search(qvec, k=max_k)
        search_ms.append((time.perf_counter() - t1) * 1000)

        row = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answerable": answerable,
            "top1_score": hits[0]["score"] if hits else None,
            "context_tokens": sum(chunk_tokens.get(h["chunk_id"], 0) for h in hits),
            "retrieved": [
                {"chunk_id": h["chunk_id"], "document_id": h["document_id"],
                 "score": h["score"], "chunk_index": h["chunk_index"]}
                for h in hits
            ],
        }

        if answerable:
            doc_flags = [any(M.doc_matches(e, h, aliases) for e in expected) for h in hits]
            row["doc_rank"] = M.first_relevant_rank(doc_flags)

            ev = evidence.get(q["id"])
            if ev:
                ev_norm = [M.normalize(e) for e in ev if e]
                row["evidence_rank"] = M.first_relevant_rank(
                    [M.chunk_has_evidence(h["text"], ev) for h in hits]
                )
                row["evidence_in_corpus"] = any(any(e in nc for e in ev_norm) for nc in norm_chunks)
        rows.append(row)

    summary = _summarize_config(config, chunks, rows, k_values, build_seconds, search_ms)
    summary["details"] = rows
    return summary, index


def _summarize_config(config, chunks, rows, k_values, build_seconds, search_ms) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    max_k = max(k_values)

    doc_ranks = [r["doc_rank"] for r in answerable]
    ev_rows = [r for r in answerable if "evidence_rank" in r]
    ev_ranks = [r["evidence_rank"] for r in ev_rows]

    summary = {
        "config": config.name,
        "chunk_size": config.chunk_size,
        "overlap": config.overlap,
        "n_chunks": len(chunks),
        "avg_chunk_tokens": round(M.mean([c["token_count"] for c in chunks]), 1),
        "index_build_seconds": round(build_seconds, 2),
        "avg_search_ms": round(M.mean(search_ms), 2),
        "avg_context_tokens": round(M.mean([r["context_tokens"] for r in rows]), 1),
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "doc_recall": {str(k): round(M.recall_at_k(doc_ranks, k), 4) for k in k_values},
        "doc_mrr": round(M.mean_reciprocal_rank(doc_ranks), 4),
        "evidence": None,
    }

    if ev_rows:
        summary["evidence"] = {
            "n_questions": len(ev_rows),
            "recall": {str(k): round(M.recall_at_k(ev_ranks, k), 4) for k in k_values},
            "mrr": round(M.mean_reciprocal_rank(ev_ranks), 4),
            "coverage": round(sum(1 for r in ev_rows if r["evidence_in_corpus"]) / len(ev_rows), 4),
        }

    # Métrica principal: evidência quando disponível; senão, documento.
    primary_key = "evidence" if summary["evidence"] else "doc"
    primary_recall = summary["evidence"]["recall"] if summary["evidence"] else summary["doc_recall"]
    summary["primary_metric"] = primary_key
    summary["primary_mrr"] = summary["evidence"]["mrr"] if summary["evidence"] else summary["doc_mrr"]
    summary["primary_recall"] = primary_recall

    # Por categoria (métrica principal)
    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in answerable}):
        cat_rows = [r for r in answerable if r["category"] == cat]
        ranks = [r["evidence_rank"] if "evidence_rank" in r else r["doc_rank"] for r in cat_rows]
        by_cat[cat] = {"n": len(cat_rows),
                       "recall": {str(k): round(M.recall_at_k(ranks, k), 4) for k in k_values},
                       "mrr": round(M.mean_reciprocal_rank(ranks), 4)}
    summary["by_category"] = by_cat

    # Separação de scores: respondíveis vs. sem resposta
    pos = [r["top1_score"] for r in answerable if r["top1_score"] is not None]
    neg = [r["top1_score"] for r in unanswerable if r["top1_score"] is not None]
    summary["score_separation"] = {
        "mean_top1_answerable": round(M.mean(pos), 4) if pos else None,
        "mean_top1_unanswerable": round(M.mean(neg), 4) if neg else None,
        "best_threshold": M.best_threshold(pos, neg),
    }
    summary["max_k"] = max_k
    return summary


def rank_configs(results: list[dict]) -> list[dict]:
    """Ordena: maior MRR principal; empate -> maior Recall@1; empate -> busca mais rápida."""
    return sorted(
        results,
        key=lambda r: (-r["primary_mrr"], -r["primary_recall"].get("1", 0.0), r["avg_search_ms"]),
    )


# ======================================================================
# Experimento de GERAÇÃO
# ======================================================================

class GenerationAborted(RuntimeError):
    """Erro fatal do LLM (modelo inexistente, chave inválida): não adianta repetir."""


_FATAL_MARKERS = ("404", "not found", "api key", "api_key", "401", "403", "permission", "invalid_api_key")


def _generate_with_retry(llm, message: str, retries: int, sleep_seconds: float) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return llm.generate(message)
        except Exception as e:  # noqa: BLE001
            last_error = e
            text = str(e).lower()
            if any(m in text for m in _FATAL_MARKERS):
                raise GenerationAborted(
                    f"Erro fatal do LLM: {e}\n"
                    "Verifique GEMINI_MODEL/OPENAI_MODEL e a chave de API no .env "
                    "(os modelos gemini-1.5-* foram descontinuados)."
                ) from e
            if attempt < retries:
                wait = max(sleep_seconds, 1.0) * (2 ** attempt)
                logger.warning("Falha na geração (tentativa %d/%d): %s. Aguardando %.0fs...",
                               attempt + 1, retries + 1, e, wait)
                time.sleep(wait)
    raise last_error  # type: ignore[misc]


NO_INFO_ANSWER = (
    "Não encontrei informações suficientes nos documentos consultados "
    "para responder com segurança.\n\n"
    "Aviso: Esta resposta foi produzida por Inteligência Artificial."
)


def run_generation_experiment(
    index,
    questions: list[dict],
    query_cache: QueryCache,
    llm,
    evidence: Optional[dict[str, list[str]]] = None,
    aliases: Optional[dict[str, list[str]]] = None,
    k: int = 5,
    sleep_seconds: float = 4.0,
    retries: int = 2,
) -> dict:
    """
    Gera respostas para todas as perguntas usando o contexto recuperado e aplica
    checagens HEURÍSTICAS automáticas. Não substitui a avaliação manual: produz
    também uma planilha com as colunas do protocolo (correta, relevante, ...).
    """
    from src.generation.llm import _build_context, _build_user_message
    from src.evaluation.generation import EvaluationRow

    evidence = evidence or {}
    aliases = aliases if aliases is not None else DEFAULT_ALIASES
    rows: list[dict] = []
    aborted: Optional[str] = None

    for i, q in enumerate(questions, start=1):
        expected = q.get("expected_documents", [])
        answerable = bool(expected)
        logger.info("[geração %d/%d] %s", i, len(questions), q["question"])

        hits = index.search(query_cache.get(q["question"]), k=k)
        source_urls = list(dict.fromkeys(h["source_url"] for h in hits if h.get("source_url")))
        error: Optional[str] = None

        if not hits:
            answer = NO_INFO_ANSWER
        else:
            message = _build_user_message(q["question"], _build_context(hits))
            try:
                answer = _generate_with_retry(llm, message, retries, sleep_seconds)
            except GenerationAborted as e:
                aborted = str(e)
                logger.error("%s", aborted)
                break
            except Exception as e:  # noqa: BLE001
                answer, error = "", str(e)
                logger.error("Erro ao gerar resposta de %s: %s", q["id"], e)

        base = EvaluationRow(question=q["question"], answer=answer or f"ERRO: {error}",
                             sources=source_urls, notas=f"ERRO: {error}" if error else "").to_dict()
        row = {"id": q["id"], "category": q["category"], "answerable": answerable, **base}

        if not error:
            expected_urls = [h["source_url"] for h in hits
                             if any(M.doc_matches(e, h, aliases) for e in expected)]
            ev = evidence.get(q["id"])
            abstained = M.is_abstention(answer)
            row["auto"] = {
                "abstained": abstained,
                "behavior_ok": abstained if not answerable else not abstained,
                "has_sources_section": M.has_sources_section(answer),
                "has_ai_warning": M.has_ai_warning(answer),
                "cites_expected_source": M.cites_url(answer, expected_urls) if answerable else None,
                "evidence_in_context": (any(M.chunk_has_evidence(h["text"], ev) for h in hits)
                                        if (answerable and ev) else None),
            }
        rows.append(row)

        if i < len(questions) and hits and sleep_seconds > 0:
            time.sleep(sleep_seconds)  # respeita limites de taxa das APIs

    result = {"k": k, "n_planned": len(questions), "n_completed": len(rows), "aborted": aborted,
              "rows": rows, "metrics": _summarize_generation(rows)}
    return result


def _summarize_generation(rows: list[dict]) -> dict:
    ok = [r for r in rows if "auto" in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    def rate(items, key):
        vals = [r["auto"][key] for r in items if r["auto"].get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None

    return {
        "n_evaluated": len(ok),
        "n_errors": len(rows) - len(ok),
        # Recusa correta nas perguntas sem resposta (teste de alucinação)
        "abstention_accuracy_unanswerable": rate(unanswerable, "abstained"),
        # Recusas indevidas em perguntas que TÊM resposta
        "false_abstention_rate_answerable": rate(answerable, "abstained"),
        "evidence_in_context_rate": rate(answerable, "evidence_in_context"),
        "cites_expected_source_rate": rate(answerable, "cites_expected_source"),
        "has_sources_section_rate": rate(ok, "has_sources_section"),
        "has_ai_warning_rate": rate(ok, "has_ai_warning"),
        "behavior_ok_rate": rate(ok, "behavior_ok"),
    }


# ======================================================================
# Orquestração
# ======================================================================

def run_experiments(
    *,
    root: Path,
    questions_file: Path,
    out_dir: Path,
    embedder,
    evidence_file: Optional[Path] = None,
    aliases_file: Optional[Path] = None,
    corpus: Optional[list[dict]] = None,
    corpus_dir: Optional[Path] = None,
    grid: Optional[list[ExperimentConfig]] = None,
    k_values: Optional[list[int]] = None,
    run_generation: bool = False,
    llm=None,
    generation_k: int = 5,
    generation_config: Optional[str] = None,
    generation_sleep: float = 4.0,
) -> dict:
    """Executa todos os experimentos, grava as saídas e devolve o dicionário de resultados."""
    root, out_dir = Path(root), Path(out_dir)
    grid = grid or make_grid()
    k_values = sorted(k_values or DEFAULT_K_VALUES)

    questions = load_questions(questions_file)
    evidence = load_evidence(evidence_file, questions)
    aliases = load_aliases(aliases_file)
    if corpus is None:
        corpus = load_corpus(root, corpus_dir=corpus_dir)
    if not corpus:
        raise RuntimeError(
            "Corpus vazio. Rode antes `python scripts/poc.py --demo` (baixa a LDB) ou "
            "`python scripts/run_pipeline.py`, ou aponte --corpus-dir para uma pasta com .html/.pdf."
        )

    query_cache = QueryCache(embedder)
    workdir = Path(tempfile.mkdtemp(prefix="docfacil_exp_"))
    indexes: dict[str, object] = {}
    results: list[dict] = []
    generation: Optional[dict] = None

    try:
        for config in grid:
            summary, index = run_retrieval_experiment(
                config, corpus, questions, embedder, query_cache, workdir,
                evidence=evidence, aliases=aliases, k_values=k_values,
            )
            results.append(summary)
            indexes[config.name] = index

        ranking = rank_configs(results)
        best = ranking[0]["config"]
        logger.info("Melhor configuração: %s (MRR %s)", best, ranking[0]["primary_mrr"])

        if run_generation:
            target = generation_config or best
            if target not in indexes:
                raise ValueError(f"Configuração '{target}' não existe. Disponíveis: {list(indexes)}")
            try:
                llm = llm or _default_llm()
                generation = run_generation_experiment(
                    indexes[target], questions, query_cache, llm, evidence=evidence,
                    aliases=aliases, k=generation_k, sleep_seconds=generation_sleep,
                )
                generation["config"] = target
            except Exception as e:  # noqa: BLE001 - não perde os resultados de recuperação
                logger.error("Avaliação de geração não executada: %s", e)
                generation = {"config": target, "error": str(e), "rows": [], "metrics": {}}
    finally:
        for index in indexes.values():
            _close_index(index)
        shutil.rmtree(workdir, ignore_errors=True)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "embedder": getattr(embedder, "model_name", type(embedder).__name__),
        "embedding_dim": getattr(embedder, "embedding_dim", None),
        "tokenizer": _tokenizer_name(),
        "questions_file": str(questions_file),
        "n_questions": len(questions),
        "n_evidence_questions": len(evidence),
        "documents": [{"document_id": d["document_id"], "chars": len(d["text"])} for d in corpus],
        "k_values": k_values,
    }
    full = {"meta": meta, "ranking": [r["config"] for r in ranking], "best": best,
            "configs": results, "generation": generation}
    write_outputs(full, out_dir)
    return full


def _default_llm():
    from src.generation.llm import _get_llm_client
    return _get_llm_client()


# ======================================================================
# Saídas
# ======================================================================

def write_outputs(full: dict, out_dir: Path) -> None:
    """Grava results.json, summary.csv, report.md e (se houver) arquivos de geração."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    _write_summary_csv(full, out_dir / "summary.csv")

    gen = full.get("generation")
    if gen and gen.get("rows"):
        with open(out_dir / "generation_results.json", "w", encoding="utf-8") as f:
            json.dump(gen, f, ensure_ascii=False, indent=2)
        _write_manual_eval_csv(gen["rows"], out_dir / "generation_manual_eval.csv")

    (out_dir / "report.md").write_text(render_report(full), encoding="utf-8")
    logger.info("Resultados gravados em: %s", out_dir)


def _write_summary_csv(full: dict, path: Path) -> None:
    k_values = full["meta"]["k_values"]
    header = ["config", "chunk_size", "overlap", "n_chunks", "avg_chunk_tokens",
              "avg_search_ms", "avg_context_tokens", "index_build_seconds", "doc_mrr"]
    header += [f"doc_recall@{k}" for k in k_values]
    header += ["evidence_mrr", "evidence_coverage"] + [f"evidence_recall@{k}" for k in k_values]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: abre bem no Excel
        w = csv.writer(f)
        w.writerow(header)
        for r in full["configs"]:
            ev = r["evidence"]
            row = [r["config"], r["chunk_size"], r["overlap"], r["n_chunks"], r["avg_chunk_tokens"],
                   r["avg_search_ms"], r["avg_context_tokens"], r["index_build_seconds"], r["doc_mrr"]]
            row += [r["doc_recall"][str(k)] for k in k_values]
            row += [ev["mrr"] if ev else "", ev["coverage"] if ev else ""]
            row += [ev["recall"][str(k)] if ev else "" for k in k_values]
            w.writerow(row)


def _write_manual_eval_csv(rows: list[dict], path: Path) -> None:
    """Planilha para a avaliação manual (colunas do protocolo de generation.py)."""
    header = ["id", "categoria", "pergunta", "resposta", "fontes",
              "auto_recusou", "auto_comportamento_ok",
              "correta", "relevante", "fundamentada", "fonte_correta", "alucinacao", "notas"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            auto = r.get("auto", {})
            w.writerow([r["id"], r["category"], r["question"], r["answer"], " | ".join(r["sources"]),
                        auto.get("abstained", ""), auto.get("behavior_ok", ""),
                        "", "", "", "", "", r.get("notas", "")])


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def render_report(full: dict) -> str:
    """Relatório em Markdown, pronto para alimentar o relatório técnico (TASK 20)."""
    meta, configs = full["meta"], full["configs"]
    ks = meta["k_values"]
    by_name = {c["config"]: c for c in configs}
    best = by_name[full["best"]]
    has_ev = best["evidence"] is not None
    n_docs = len(meta["documents"])
    L: list[str] = []

    L.append("# Relatório de experimentos de avaliação (TASK 17)\n")
    L.append(f"Gerado em {meta['generated_at']}.\n")

    L.append("## Resumo\n")
    L.append(f"- **Corpus:** {n_docs} documento(s) — " + ", ".join(f"`{d['document_id']}`" for d in meta["documents"]))
    L.append(f"- **Perguntas:** {best['n_questions']} ({best['n_answerable']} com resposta, "
             f"{best['n_unanswerable']} sem resposta; {meta['n_evidence_questions']} com evidência anotada)")
    L.append(f"- **Embedder:** `{meta['embedder']}` ({meta['embedding_dim']} dimensões)")
    L.append(f"- **Tokenizador do chunking:** {meta['tokenizer']}")
    L.append(f"- **Métrica principal:** {'MRR por evidência (nível de trecho)' if has_ev else 'MRR por documento'}")
    L.append(f"- **Melhor configuração:** **`{best['config']}`** — chunk de {best['chunk_size']} tokens, "
             f"sobreposição {best['overlap']}, MRR {best['primary_mrr']:.3f}, "
             f"Recall@1 {_pct(best['primary_recall'].get('1'))}\n")

    # Comparação
    L.append("## Comparação de configurações\n")
    head = ["Config", "Chunks", "Tokens/chunk"] + [f"Doc R@{k}" for k in ks]
    if has_ev:
        head += [f"Evid. R@{k}" for k in ks] + ["Evid. MRR", "Cobertura"]
    head += [f"Contexto (Top-{max(ks)}, tokens)", "Busca (ms)"]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in [by_name[n] for n in full["ranking"]]:
        cells = [f"`{r['config']}`" + (" ⭐" if r["config"] == full["best"] else ""),
                 str(r["n_chunks"]), str(r["avg_chunk_tokens"])]
        cells += [_pct(r["doc_recall"][str(k)]) for k in ks]
        if has_ev and r["evidence"]:
            cells += [_pct(r["evidence"]["recall"][str(k)]) for k in ks]
            cells += [f"{r['evidence']['mrr']:.3f}", _pct(r["evidence"]["coverage"])]
        cells += [f"{r['avg_context_tokens']:.0f}", f"{r['avg_search_ms']:.1f}"]
        L.append("| " + " | ".join(cells) + " |")
    L.append("\n*Ordenado do melhor para o pior. Cobertura = % das perguntas cuja evidência existe inteira em algum chunk. "
             "Contexto = tokens médios enviados ao LLM no Top-K máximo (custo e risco de ruído).*\n")

    # Categorias
    L.append(f"## Desempenho por categoria — `{best['config']}`\n")
    L.append("| Categoria | N | " + " | ".join(f"R@{k}" for k in ks) + " | MRR |")
    L.append("|" + "|".join(["---"] * (len(ks) + 3)) + "|")
    for cat, v in best["by_category"].items():
        L.append(f"| {cat} | {v['n']} | " + " | ".join(_pct(v["recall"][str(k)]) for k in ks) + f" | {v['mrr']:.3f} |")
    L.append("")

    # Sem resposta
    sep = best["score_separation"]
    L.append("## Perguntas sem resposta: o score de similaridade separa?\n")
    if sep["best_threshold"]:
        bt = sep["best_threshold"]
        L.append(f"- Score médio do Top-1 em perguntas **com** resposta: {sep['mean_top1_answerable']}")
        L.append(f"- Score médio do Top-1 em perguntas **sem** resposta: {sep['mean_top1_unanswerable']}")
        L.append(f"- Melhor limiar de corte: **{bt['threshold']}** "
                 f"(acurácia balanceada {_pct(bt['balanced_accuracy'])}; "
                 f"acerta {_pct(bt['tpr'])} das respondíveis e {_pct(bt['tnr'])} das não respondíveis)")
        L.append(f"- Base pequena ({bt['n_answerable']} vs {bt['n_unanswerable']} perguntas): "
                 "trate o limiar como indicativo, não como valor definitivo.\n")
    else:
        L.append("Sem perguntas sem resposta (ou sem resultados) para calcular um limiar.\n")

    # Falhas
    key = "evidence_rank" if has_ev else "doc_rank"
    misses = [r for r in best["details"] if r["answerable"] and r.get(key, 0) is None]
    L.append(f"## Falhas de recuperação — `{best['config']}` (nenhum acerto no Top-{max(ks)})\n")
    if misses:
        for r in misses:
            top = r["retrieved"][0] if r["retrieved"] else None
            top_txt = f"Top-1: chunk {top['chunk_index']} (score {top['score']})" if top else "sem resultados"
            hint = ""
            if has_ev and r.get("evidence_in_corpus") is False:
                hint = " ⚠️ a evidência não existe inteira em nenhum chunk (corte de fronteira ou evidência incorreta)"
            L.append(f"- `{r['id']}` [{r['category']}] {r['question']} — {top_txt}{hint}")
    else:
        L.append(f"Nenhuma: todas as perguntas tiveram acerto no Top-{max(ks)}.")
    L.append("")

    # Geração
    gen = full.get("generation")
    if gen:
        L.append(f"## Avaliação de geração — `{gen.get('config')}`\n")
        if gen.get("error"):
            L.append(f"⚠️ **Não executada:** {gen['error']}\n")
        else:
            if gen.get("aborted"):
                L.append(f"⚠️ **Interrompida** após {gen['n_completed']}/{gen['n_planned']} perguntas: {gen['aborted']}\n")
            m = gen["metrics"]
            L.append("Checagens **heurísticas** automáticas (não substituem a avaliação manual em "
                     "`generation_manual_eval.csv`):\n")
            L.append("| Métrica | Valor |")
            L.append("|---|---|")
            L.append(f"| Perguntas avaliadas / com erro | {m['n_evaluated']} / {m['n_errors']} |")
            L.append(f"| Recusa correta em perguntas sem resposta | {_pct(m['abstention_accuracy_unanswerable'])} |")
            L.append(f"| Recusa indevida em perguntas com resposta | {_pct(m['false_abstention_rate_answerable'])} |")
            L.append(f"| Evidência presente no contexto enviado ao LLM | {_pct(m['evidence_in_context_rate'])} |")
            L.append(f"| Cita a URL do documento esperado | {_pct(m['cites_expected_source_rate'])} |")
            L.append(f"| Traz seção \"Fontes\" | {_pct(m['has_sources_section_rate'])} |")
            L.append(f"| Traz aviso de IA | {_pct(m['has_ai_warning_rate'])} |")
            L.append("")

    # Avisos
    L.append("## Avisos e limitações\n")
    if n_docs == 1:
        L.append("- **Corpus com 1 documento:** o Recall por documento é trivialmente alto em qualquer "
                 "configuração. Use as colunas de **evidência** para comparar; para validar de verdade a "
                 "recuperação entre documentos, indexe mais documentos (`run_pipeline.py`).")
    if not has_ev:
        L.append("- **Sem evidências anotadas:** só há métricas por documento (pouco discriminativas). "
                 "Veja `data/questions/evidence_ldb.json`.")
    elif any(r["evidence"] and r["evidence"]["coverage"] < 1 for r in configs):
        L.append("- **Cobertura < 100%** em alguma configuração: em parte das perguntas a evidência foi cortada na "
                 "fronteira entre chunks (penaliza chunks pequenos/sem sobreposição) ou está incorreta. "
                 "Valide com `python scripts/run_experiments.py --check-evidence`.")
    if "hashing" in str(meta["embedder"]).lower():
        L.append("- **Embedder offline (lexical):** linha de base sem semântica. Estes números NÃO representam "
                 "o `multilingual-e5-large`; rode sem `--embedder hashing` para o resultado real.")
    if "aproximado" in meta["tokenizer"]:
        L.append("- **Tokenizador aproximado:** o `tiktoken` não pôde carregar seu vocabulário (rede?); "
                 "tamanhos de chunk são aproximados.")
    L.append("- **Viés a favor de chunks grandes:** um chunk maior contém mais texto e, por isso, acerta a evidência "
             "com mais facilidade — mas leva mais tokens de contexto ao LLM, o que encarece e pode diluir a "
             "resposta. Decida olhando Recall **e** a coluna de contexto, não só o MRR.")
    L.append("- Todas as perguntas foram elaboradas sobre um único documento (LDB); generalização para outros "
             "documentos não foi medida.")
    L.append("- Evidências são heurísticas de correspondência textual: uma resposta pode estar correta mesmo sem "
             "conter literalmente a evidência (e vice-versa).")
    L.append("")
    return "\n".join(L)
