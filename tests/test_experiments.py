"""
tests/test_experiments.py

Testes da TASK 17 (experimentos de avaliação): métricas, embedder offline,
fallback do chunker, carregamento de dados e execução ponta a ponta.

Rodam 100% offline: usam HashingEmbedder, um LLM falso e o documento sintético
de tests/fixtures (nenhum modelo é baixado, nenhuma API é chamada).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"
QUESTIONS = ROOT / "data" / "questions" / "test_questions.json"
EVIDENCE = ROOT / "data" / "questions" / "evidence_ldb.json"
ALIASES = ROOT / "data" / "questions" / "document_aliases.json"


# ============================================================
# metrics.py
# ============================================================

class TestMetrics:
    def test_normalize_remove_acentos_e_caixa(self):
        from src.evaluation.metrics import normalize
        assert normalize("  Educação   BÁSICA ") == "educacao basica"
        assert normalize("") == ""

    def test_doc_matches_id_exato(self):
        from src.evaluation.metrics import doc_matches
        assert doc_matches("poc_ldb_l9394", {"document_id": "poc_ldb_l9394"})

    def test_doc_matches_por_alias_na_url_ou_id(self):
        from src.evaluation.metrics import doc_matches
        aliases = {"poc_ldb_l9394": ["l9394"]}
        chunk = {"document_id": "ccivil_03_leis_l9394_ab12", "source_url": "https://x/l9394.htm"}
        assert doc_matches("poc_ldb_l9394", chunk, aliases)
        assert not doc_matches("poc_ldb_l9394", {"document_id": "outro", "source_url": "https://x/l8666"}, aliases)

    def test_doc_matches_sem_alias_nao_casa(self):
        from src.evaluation.metrics import doc_matches
        assert not doc_matches("poc_ldb_l9394", {"document_id": "outro"})

    def test_chunk_has_evidence_alternativas(self):
        from src.evaluation.metrics import chunk_has_evidence
        text = "A carga horária mínima será de OITOCENTAS horas."
        assert chunk_has_evidence(text, ["duzentos dias", "oitocentas horas"])
        assert not chunk_has_evidence(text, ["duzentos dias"])
        assert not chunk_has_evidence(text, [])

    def test_first_relevant_rank(self):
        from src.evaluation.metrics import first_relevant_rank
        assert first_relevant_rank([False, False, True, True]) == 3
        assert first_relevant_rank([False, False]) is None
        assert first_relevant_rank([]) is None

    def test_recall_at_k_e_mrr(self):
        from src.evaluation.metrics import recall_at_k, mean_reciprocal_rank
        ranks = [1, 2, None, 5]
        assert recall_at_k(ranks, 1) == 0.25
        assert recall_at_k(ranks, 3) == 0.5
        assert recall_at_k(ranks, 5) == 0.75
        assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)

    def test_agregacao_vazia_retorna_zero(self):
        from src.evaluation.metrics import recall_at_k, mean_reciprocal_rank
        assert recall_at_k([], 3) == 0.0
        assert mean_reciprocal_rank([]) == 0.0

    def test_best_threshold_separacao_perfeita(self):
        from src.evaluation.metrics import best_threshold
        r = best_threshold([0.8, 0.9, 0.85], [0.4, 0.5])
        assert r["balanced_accuracy"] == 1.0
        assert 0.5 < r["threshold"] < 0.8

    def test_best_threshold_empate_prefere_limiar_mais_alto(self):
        """Limiares 0.2 e 0.6 empatam em acurácia balanceada (0.75): vence o mais conservador."""
        from src.evaluation.metrics import best_threshold
        r = best_threshold([0.9, 0.3], [0.3, 0.1])
        assert r["balanced_accuracy"] == 0.75
        assert r["threshold"] == 0.6

    def test_best_threshold_sem_um_dos_grupos(self):
        from src.evaluation.metrics import best_threshold
        assert best_threshold([], [0.5]) is None
        assert best_threshold([0.5], []) is None

    def test_best_threshold_sobreposicao(self):
        from src.evaluation.metrics import best_threshold
        r = best_threshold([0.6, 0.7, 0.55], [0.58, 0.5])
        assert 0.5 <= r["balanced_accuracy"] <= 1.0
        assert r["n_answerable"] == 3 and r["n_unanswerable"] == 2

    def test_is_abstention(self):
        from src.evaluation.metrics import is_abstention
        assert is_abstention("Não foi possível encontrar informação suficiente nos documentos.")
        assert is_abstention("Não encontrei informações suficientes.")
        assert not is_abstention("O ensino fundamental tem duração de 9 anos.")

    def test_has_ai_warning_e_fontes_e_url(self):
        from src.evaluation.metrics import has_ai_warning, has_sources_section, cites_url
        ans = "Resposta: x\n\nFontes:\n- Lei\n  http://a/b\n\nAviso: Inteligência Artificial"
        assert has_ai_warning(ans) and has_sources_section(ans)
        assert cites_url(ans, ["http://a/b"])
        assert not cites_url(ans, ["http://outra"])
        assert not has_ai_warning("sem aviso")


# ============================================================
# embeddings/offline.py
# ============================================================

class TestHashingEmbedder:
    def test_deterministico_entre_instancias(self):
        from src.embeddings.offline import HashingEmbedder
        a = HashingEmbedder().embed_query("duzentos dias letivos")
        b = HashingEmbedder().embed_query("duzentos dias letivos")
        assert np.array_equal(a, b)

    def test_vetores_normalizados(self):
        from src.embeddings.offline import HashingEmbedder
        e = HashingEmbedder()
        vecs = e.embed_texts(["Art. 1º A educação abrange", "Art. 2º dever da família"])
        assert vecs.shape == (2, e.embedding_dim)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_texto_relacionado_fica_mais_proximo(self):
        from src.embeddings.offline import HashingEmbedder
        e = HashingEmbedder()
        docs = e.embed_texts([
            "A carga horária mínima anual será de oitocentas horas em duzentos dias letivos.",
            "Os Municípios oferecerão a educação infantil em creches e pré-escolas.",
        ])
        q = e.embed_query("Quantos dias letivos mínimos?")
        assert (docs @ q)[0] > (docs @ q)[1]

    def test_lista_vazia(self):
        from src.embeddings.offline import HashingEmbedder
        assert HashingEmbedder().embed_texts([]).size == 0

    def test_texto_so_stopwords_nao_quebra(self):
        from src.embeddings.offline import HashingEmbedder
        v = HashingEmbedder().embed_query("de da o a")
        assert v.shape[0] == 1024 and not np.isnan(v).any()


# ============================================================
# processing/chunker.py — fallback offline
# ============================================================

class TestChunkerFallback:
    def test_regex_encoding_e_reversivel(self):
        from src.processing.chunker import _RegexEncoding
        enc = _RegexEncoding()
        text = "Art. 1º  Esta lei\n\n§ 2º — vale, sim!  \n"
        assert enc.decode(enc.encode(text)) == text

    def test_chunker_funciona_quando_tiktoken_falha(self, monkeypatch):
        import tiktoken
        from src.processing import chunker as ch

        def boom(name):
            raise ConnectionError("sem rede")

        monkeypatch.setattr(tiktoken, "get_encoding", boom)
        monkeypatch.setattr(ch, "_ENCODING_CACHE", {})
        text = " ".join(["palavra"] * 300)
        chunks = ch.chunk_document(text, "doc", {"source_url": "u"}, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(c["document_id"] == "doc" for c in chunks)

    def test_chunks_cobrem_todo_o_texto_sem_overlap(self, monkeypatch):
        import tiktoken
        from src.processing import chunker as ch
        monkeypatch.setattr(tiktoken, "get_encoding", lambda n: (_ for _ in ()).throw(OSError("x")))
        monkeypatch.setattr(ch, "_ENCODING_CACHE", {})
        text = "um dois três quatro cinco seis sete oito nove dez " * 20
        chunks = ch.chunk_document(text, "d", {}, chunk_size=50, overlap=0)
        assert "".join(c["text"] for c in chunks) == text


# ============================================================
# experiments.py — carregamento
# ============================================================

class TestLoaders:
    def test_make_grid_ignora_combinacoes_invalidas(self):
        from src.evaluation.experiments import make_grid
        names = [c.name for c in make_grid([100, 500], [0, 100, 200])]
        assert "cs100_ov100" not in names and "cs100_ov0" in names
        assert "cs500_ov200" in names

    def test_config_overlap_invalido_levanta(self):
        from src.evaluation.experiments import ExperimentConfig
        with pytest.raises(ValueError):
            ExperimentConfig(100, 100)

    def test_load_questions_garante_campos(self, tmp_path):
        from src.evaluation.experiments import load_questions
        f = tmp_path / "q.json"
        f.write_text(json.dumps([{"question": "x?"}]), encoding="utf-8")
        q = load_questions(f)[0]
        assert q["id"] == "q01" and q["expected_documents"] == [] and q["category"] == "sem_categoria"

    def test_load_evidence_prioriza_campo_inline_e_ignora_comentario(self, tmp_path):
        from src.evaluation.experiments import load_evidence
        side = tmp_path / "ev.json"
        side.write_text(json.dumps({"_comment": "x", "q01": ["do sidecar"], "q02": ["só sidecar"]}),
                        encoding="utf-8")
        questions = [{"id": "q01", "evidence": ["inline"]}, {"id": "q02"}, {"id": "q03"}]
        ev = load_evidence(side, questions)
        assert ev == {"q01": ["inline"], "q02": ["só sidecar"]}

    def test_load_evidence_arquivo_ausente(self):
        from src.evaluation.experiments import load_evidence
        assert load_evidence(Path("/nao/existe.json"), [{"id": "q01"}]) == {}

    def test_load_aliases_mescla_com_padrao(self, tmp_path):
        from src.evaluation.experiments import load_aliases
        f = tmp_path / "a.json"
        f.write_text(json.dumps({"_comment": "x", "poc_ldb_l9394": ["ldb"], "outro": ["o"]}), encoding="utf-8")
        a = load_aliases(f)
        assert set(a["poc_ldb_l9394"]) == {"l9394", "ldb"} and a["outro"] == ["o"]

    def test_arquivos_reais_do_projeto_sao_validos(self):
        from src.evaluation.experiments import load_questions, load_evidence, load_aliases
        questions = load_questions(QUESTIONS)
        evidence = load_evidence(EVIDENCE, questions)
        assert len(questions) == 20
        # toda evidência aponta para pergunta que existe e TEM resposta
        by_id = {q["id"]: q for q in questions}
        assert set(evidence) <= set(by_id)
        assert all(by_id[i]["expected_documents"] for i in evidence)
        assert "poc_ldb_l9394" in load_aliases(ALIASES)


class TestCorpus:
    def test_load_corpus_de_pasta_usa_alias_da_poc(self):
        from src.evaluation.experiments import load_corpus
        corpus = load_corpus(ROOT, corpus_dir=FIXTURES)
        assert [d["document_id"] for d in corpus] == ["poc_ldb_l9394"]
        assert "duzentos dias" in corpus[0]["text"]

    def test_load_corpus_fallback_data_raw_html(self, tmp_path):
        from src.evaluation.experiments import load_corpus
        raw = tmp_path / "data" / "raw" / "html"
        raw.mkdir(parents=True)
        (raw / "poc_ldb_demo.html").write_text(
            (FIXTURES / "poc_ldb_demo.html").read_text(encoding="utf-8"), encoding="utf-8")
        corpus = load_corpus(tmp_path)
        assert corpus and corpus[0]["document_id"] == "poc_ldb_l9394"

    def test_load_corpus_via_documents_jsonl(self, tmp_path):
        from src.evaluation.experiments import load_corpus
        raw = tmp_path / "data" / "raw" / "html"
        raw.mkdir(parents=True)
        (raw / "lei_x.html").write_text(
            "<html><body><h1>Lei X</h1><p>Art. 1º Texto da lei X sobre educação.</p></body></html>",
            encoding="utf-8")
        proc = tmp_path / "data" / "processed"
        proc.mkdir(parents=True)
        (proc / "documents.jsonl").write_text(json.dumps({
            "document_id": "lei_x_abc", "title": "Lei X", "source_url": "https://exemplo/lei_x",
            "file_path": "data/raw/html/lei_x.html", "content_type": "html"}) + "\n", encoding="utf-8")
        corpus = load_corpus(tmp_path)
        assert len(corpus) == 1  # não duplica o mesmo arquivo da varredura de data/raw
        assert corpus[0]["document_id"] == "lei_x_abc"
        assert corpus[0]["meta"]["source_url"] == "https://exemplo/lei_x"

    def test_load_corpus_vazio(self, tmp_path):
        from src.evaluation.experiments import load_corpus
        assert load_corpus(tmp_path) == []

    def test_check_evidence_detecta_faltantes(self):
        from src.evaluation.experiments import load_corpus, check_evidence
        corpus = load_corpus(ROOT, corpus_dir=FIXTURES)
        questions = [{"id": "q01", "question": "a"}, {"id": "q02", "question": "b"}]
        ev = {"q01": ["duzentos dias"], "q02": ["frase que não existe no texto"]}
        rep = {r["id"]: r["ok"] for r in check_evidence(corpus, questions, ev)}
        assert rep == {"q01": True, "q02": False}

    def test_evidencias_do_projeto_existem_no_fixture(self):
        """O fixture contém as frases das evidências: garante que o teste de fumaça é válido."""
        from src.evaluation.experiments import load_corpus, load_questions, load_evidence, check_evidence
        corpus = load_corpus(ROOT, corpus_dir=FIXTURES)
        questions = load_questions(QUESTIONS)
        rep = check_evidence(corpus, questions, load_evidence(EVIDENCE, questions))
        assert rep and all(r["ok"] for r in rep), [r["id"] for r in rep if not r["ok"]]


# ============================================================
# experiments.py — execução ponta a ponta
# ============================================================

chromadb = pytest.importorskip("chromadb")


class FakeLLM:
    """LLM falso: recusa perguntas fora do escopo, responde com fontes nas demais."""

    def __init__(self):
        self.calls = 0

    def generate(self, message: str) -> str:
        self.calls += 1
        question = message.split("CONTEXTO")[0].lower()
        if any(w in question for w in ("imposto", "salario", "licitacao")):
            return ("Não encontrei informação suficiente nos documentos consultados.\n\n"
                    "Aviso: produzida por Inteligência Artificial.")
        return ("Resposta: conforme os trechos.\n\nFontes:\n- Lei 9.394\n  file:///x\n\n"
                "Aviso: Esta resposta foi produzida por Inteligência Artificial.")


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    from src.embeddings.offline import HashingEmbedder
    from src.evaluation import experiments as X

    out = tmp_path_factory.mktemp("exp_out")
    full = X.run_experiments(
        root=ROOT, questions_file=QUESTIONS, evidence_file=EVIDENCE, aliases_file=ALIASES,
        out_dir=out, embedder=HashingEmbedder(), corpus_dir=FIXTURES,
        grid=X.make_grid([250, 500], [0, 100]), k_values=[1, 3, 5],
        run_generation=True, llm=FakeLLM(), generation_sleep=0,
    )
    return full, out


class TestEndToEnd:
    def test_gera_todos_os_arquivos(self, run_result):
        _, out = run_result
        for name in ["results.json", "summary.csv", "report.md",
                     "generation_results.json", "generation_manual_eval.csv"]:
            assert (out / name).exists(), name

    def test_uma_linha_por_configuracao(self, run_result):
        full, out = run_result
        assert len(full["configs"]) == 4
        with open(out / "summary.csv", encoding="utf-8-sig") as f:
            assert len(list(csv.DictReader(f))) == 4

    def test_ranking_ordenado_e_melhor_e_o_primeiro(self, run_result):
        full, _ = run_result
        by = {c["config"]: c for c in full["configs"]}
        mrrs = [by[n]["primary_mrr"] for n in full["ranking"]]
        assert mrrs == sorted(mrrs, reverse=True)
        assert full["best"] == full["ranking"][0]

    def test_metricas_no_intervalo_valido(self, run_result):
        full, _ = run_result
        for c in full["configs"]:
            assert 0.0 <= c["doc_mrr"] <= 1.0 and 0.0 <= c["evidence"]["mrr"] <= 1.0
            for k in ("1", "3", "5"):
                assert 0.0 <= c["doc_recall"][k] <= 1.0
                assert 0.0 <= c["evidence"]["recall"][k] <= 1.0
            # Recall@K é monotônico em K
            assert c["evidence"]["recall"]["1"] <= c["evidence"]["recall"]["3"] <= c["evidence"]["recall"]["5"]

    def test_corpus_de_um_documento_tem_doc_recall_trivial(self, run_result):
        """Documenta o motivo de existir a métrica por evidência."""
        full, _ = run_result
        assert all(c["doc_recall"]["1"] == 1.0 for c in full["configs"])

    def test_contagem_de_perguntas(self, run_result):
        full, _ = run_result
        c = full["configs"][0]
        assert (c["n_questions"], c["n_answerable"], c["n_unanswerable"]) == (20, 17, 3)
        assert full["meta"]["n_evidence_questions"] == 17

    def test_cobertura_total_no_fixture(self, run_result):
        full, _ = run_result
        assert all(c["evidence"]["coverage"] == 1.0 for c in full["configs"])

    def test_limiar_de_perguntas_sem_resposta_calculado(self, run_result):
        full, _ = run_result
        bt = full["configs"][0]["score_separation"]["best_threshold"]
        assert bt is not None and 0.0 <= bt["balanced_accuracy"] <= 1.0

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="ChromaDB mantém arquivos travados no Windows; a limpeza é best-effort lá")
    def test_diretorio_de_trabalho_temporario_e_removido(self, run_result):
        import tempfile
        leftovers = list(Path(tempfile.gettempdir()).glob("docfacil_exp_*"))
        assert leftovers == []

    def test_nao_toca_o_vectorstore_do_app(self, tmp_path):
        """Os índices dos experimentos são isolados: ./vectorstore (usado pelo app) fica intacto."""
        from src.embeddings.offline import HashingEmbedder
        from src.evaluation import experiments as X

        def snapshot():
            vs = ROOT / "vectorstore"
            if not vs.exists():
                return None
            return sorted((str(p.relative_to(vs)), p.stat().st_mtime_ns, p.stat().st_size)
                          for p in vs.rglob("*") if p.is_file())

        before = snapshot()
        X.run_experiments(root=ROOT, questions_file=QUESTIONS, evidence_file=EVIDENCE, aliases_file=ALIASES,
                          out_dir=tmp_path, embedder=HashingEmbedder(), corpus_dir=FIXTURES,
                          grid=X.make_grid([500], [100]))
        assert snapshot() == before

    def test_relatorio_tem_secoes_e_avisos(self, run_result):
        _, out = run_result
        md = (out / "report.md").read_text(encoding="utf-8")
        for section in ["## Resumo", "## Comparação de configurações", "## Desempenho por categoria",
                        "## Perguntas sem resposta", "## Avaliação de geração", "## Avisos e limitações"]:
            assert section in md, section
        assert "Embedder offline" in md            # aviso do hashing
        assert "Corpus com 1 documento" in md      # aviso de recall trivial
        assert "Viés a favor de chunks grandes" in md


class TestGeracao:
    def test_geracao_avaliou_todas_as_perguntas(self, run_result):
        full, _ = run_result
        gen = full["generation"]
        assert gen["n_completed"] == gen["n_planned"] == 20 and gen["aborted"] is None

    def test_recusa_correta_nas_perguntas_sem_resposta(self, run_result):
        full, _ = run_result
        m = full["generation"]["metrics"]
        assert m["abstention_accuracy_unanswerable"] == 1.0
        assert m["false_abstention_rate_answerable"] == 0.0
        assert m["has_ai_warning_rate"] == 1.0
        # O LLM falso só inclui "Fontes" nas 17 respostas reais, não nas 3 recusas
        assert m["has_sources_section_rate"] == pytest.approx(17 / 20)

    def test_planilha_manual_tem_colunas_do_protocolo(self, run_result):
        _, out = run_result
        with open(out / "generation_manual_eval.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20
        for col in ["correta", "relevante", "fundamentada", "fonte_correta", "alucinacao"]:
            assert col in rows[0] and rows[0][col] == ""   # a preencher manualmente

    def test_geracao_usa_a_melhor_configuracao(self, run_result):
        full, _ = run_result
        assert full["generation"]["config"] == full["best"]


class TestFluxoDoColetor:
    def test_ids_do_coletor_casam_com_perguntas_via_alias(self, tmp_path):
        """
        Cenário real: run_pipeline.py gera document_id = slug + hash da URL, mas as
        perguntas esperam 'poc_ldb_l9394'. Sem os aliases o Recall por documento seria 0.
        """
        from src.embeddings.offline import HashingEmbedder
        from src.evaluation import experiments as X

        root = tmp_path / "proj"
        (root / "data" / "raw" / "html").mkdir(parents=True)
        (root / "data" / "processed").mkdir(parents=True)
        (root / "data" / "raw" / "html" / "l9394.htm_ab12cd34.html").write_text(
            (FIXTURES / "poc_ldb_demo.html").read_text(encoding="utf-8"), encoding="utf-8")
        (root / "data" / "processed" / "documents.jsonl").write_text(json.dumps({
            "document_id": "ccivil_03_leis_l9394.htm_ab12cd34",
            "source_url": "https://www.planalto.gov.br/ccivil_03/leis/l9394.htm",
            "file_path": "data/raw/html/l9394.htm_ab12cd34.html", "content_type": "html"}) + "\n",
            encoding="utf-8")

        kw = dict(root=root, questions_file=QUESTIONS, evidence_file=EVIDENCE, out_dir=tmp_path / "out",
                  embedder=HashingEmbedder(), grid=X.make_grid([500], [100]))

        com_alias = X.run_experiments(aliases_file=ALIASES, **kw)["configs"][0]
        assert com_alias["doc_recall"]["5"] == 1.0
        assert com_alias["evidence"]["coverage"] == 1.0

        # contraprova: sem o arquivo de aliases, só o alias padrão embutido (l9394) já resolve;
        # com um alias errado, o Recall por documento cai a zero — prova que o casamento é real.
        bad = tmp_path / "bad_aliases.json"
        bad.write_text(json.dumps({"poc_ldb_l9394": ["xxxx"]}), encoding="utf-8")
        import src.evaluation.experiments as mod
        original = dict(mod.DEFAULT_ALIASES)
        mod.DEFAULT_ALIASES.clear()
        try:
            sem_alias = X.run_experiments(aliases_file=bad, **{**kw, "out_dir": tmp_path / "out2"})["configs"][0]
        finally:
            mod.DEFAULT_ALIASES.update(original)
        assert sem_alias["doc_recall"]["5"] == 0.0


class TestGeracaoErros:
    def _setup(self, tmp_path):
        from src.embeddings.offline import HashingEmbedder
        from src.evaluation import experiments as X
        return X, dict(root=ROOT, questions_file=QUESTIONS, evidence_file=EVIDENCE, aliases_file=ALIASES,
                       out_dir=tmp_path, embedder=HashingEmbedder(), corpus_dir=FIXTURES,
                       grid=X.make_grid([500], [100]), run_generation=True, generation_sleep=0)

    def test_erro_fatal_interrompe_e_preserva_recuperacao(self, tmp_path):
        class Fatal:
            def generate(self, m):
                raise RuntimeError("404 models/gemini-1.5-flash is not found")

        X, kw = self._setup(tmp_path)
        full = X.run_experiments(llm=Fatal(), **kw)
        gen = full["generation"]
        assert gen["aborted"] and "gemini" in gen["aborted"].lower()
        assert gen["n_completed"] == 0           # parou na primeira pergunta, sem 20 retries
        assert full["configs"][0]["n_questions"] == 20  # recuperação preservada
        assert "Interrompida" in (tmp_path / "report.md").read_text(encoding="utf-8")

    def test_erro_transitorio_e_repetido(self, monkeypatch):
        from src.evaluation import experiments as X
        monkeypatch.setattr(X.time, "sleep", lambda s: None)

        class Flaky:
            n = 0

            def generate(self, m):
                Flaky.n += 1
                if Flaky.n < 3:
                    raise TimeoutError("timeout")
                return "ok"

        assert X._generate_with_retry(Flaky(), "msg", retries=2, sleep_seconds=0) == "ok"
        assert Flaky.n == 3

    def test_erro_transitorio_persistente_levanta(self, monkeypatch):
        from src.evaluation import experiments as X
        monkeypatch.setattr(X.time, "sleep", lambda s: None)

        class Always:
            def generate(self, m):
                raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            X._generate_with_retry(Always(), "msg", retries=1, sleep_seconds=0)

    def test_sem_llm_configurado_nao_perde_recuperacao(self, tmp_path, monkeypatch):
        from src.evaluation import experiments as X
        _, kw = self._setup(tmp_path)

        def no_llm():
            raise ValueError("GOOGLE_API_KEY não encontrada no .env")

        monkeypatch.setattr(X, "_default_llm", no_llm)
        full = X.run_experiments(llm=None, **kw)
        assert "GOOGLE_API_KEY" in full["generation"]["error"]
        assert (tmp_path / "report.md").exists()

    def test_corpus_vazio_levanta_erro_claro(self, tmp_path):
        from src.embeddings.offline import HashingEmbedder
        from src.evaluation import experiments as X
        with pytest.raises(RuntimeError, match="Corpus vazio"):
            X.run_experiments(root=tmp_path, questions_file=QUESTIONS, out_dir=tmp_path / "o",
                              embedder=HashingEmbedder(), grid=X.make_grid([500], [100]))

    def test_config_de_geracao_inexistente(self, tmp_path):
        X, kw = self._setup(tmp_path)
        kw["generation_config"] = "cs999_ov0"
        with pytest.raises(ValueError, match="não existe"):
            X.run_experiments(llm=object(), **kw)
