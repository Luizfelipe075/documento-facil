"""
src/evaluation/metrics.py

Métricas puras (sem I/O, sem modelos) usadas nos experimentos de avaliação.
Ficam separadas para serem testáveis rapidamente e reutilizáveis.

Dois níveis de relevância:

1. DOCUMENTO — o chunk pertence a um documento esperado (document_id).
   Problema: se o índice tem um único documento (ex.: só a LDB), todo chunk
   é "relevante" e Recall@K vira 100% para qualquer configuração.

2. EVIDÊNCIA (nível de trecho) — o chunk contém um dos textos-evidência
   da pergunta (ex.: "duzentos dias"). É o que realmente diferencia
   configurações de chunking/embedding.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

# Frases que indicam que o modelo se recusou a responder (abstenção).
# Comparação feita sobre texto normalizado (sem acento, minúsculo).
ABSTENTION_PATTERNS = [
    "nao encontrei informac",
    "nao foi possivel encontrar",
    "nao ha informac",
    "nao consta",
    "nao foram encontrad",
    "contexto nao e suficiente",
    "contexto nao eh suficiente",
    "informacao suficiente",
    "informacoes suficientes",
    "nao trata de",
    "nao aborda",
    "nao menciona",
    "nao dispoe",
]


def normalize(text: str) -> str:
    """Minúsculas, sem acentos, espaços colapsados. Para comparação tolerante."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents.lower()).strip()


# ----------------------------------------------------------------------
# Relevância
# ----------------------------------------------------------------------

def doc_matches(
    expected: str,
    chunk: dict,
    aliases: Optional[dict[str, list[str]]] = None,
) -> bool:
    """
    O chunk pertence ao documento esperado?

    Casa por document_id exato, ou por qualquer alias presente no
    document_id / source_url do chunk. Os aliases resolvem o descompasso entre
    o ID simbólico das perguntas ("poc_ldb_l9394") e os IDs gerados pelo
    coletor (slug + hash da URL).
    """
    doc_id = chunk.get("document_id", "") or ""
    if expected == doc_id:
        return True
    haystack = f"{doc_id} {chunk.get('source_url', '') or ''}".lower()
    for alias in (aliases or {}).get(expected, []):
        if alias.lower() in haystack:
            return True
    return False


def chunk_has_evidence(text: str, evidence: Sequence[str]) -> bool:
    """True se o texto contém QUALQUER uma das evidências (alternativas)."""
    norm = normalize(text)
    return any(normalize(e) in norm for e in evidence if e)


def first_relevant_rank(flags: Sequence[bool]) -> Optional[int]:
    """Posição (1-based) do primeiro True em `flags`; None se não houver."""
    for i, flag in enumerate(flags, start=1):
        if flag:
            return i
    return None


# ----------------------------------------------------------------------
# Agregação
# ----------------------------------------------------------------------

def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """Fração de perguntas cujo primeiro acerto está entre as K primeiras posições."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> float:
    """MRR: média de 1/posição do primeiro acerto (0 quando não há acerto)."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ----------------------------------------------------------------------
# Perguntas sem resposta: limiar de similaridade
# ----------------------------------------------------------------------

def best_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> Optional[dict]:
    """
    Procura o limiar de similaridade que melhor separa perguntas
    respondíveis (positivas) das não respondíveis (negativas), usando o
    score do Top-1. Regra: score >= limiar => "há resposta nos documentos".

    Critério: maior acurácia balanceada; em empate, prefere o limiar mais
    alto (mais conservador — evitar alucinação é prioridade no projeto).

    Retorna None se faltar algum dos dois grupos.
    """
    if not positive_scores or not negative_scores:
        return None

    all_scores = sorted(set(positive_scores) | set(negative_scores))
    # Candidatos: pontos médios entre scores vizinhos + extremos.
    candidates = [all_scores[0] - 1e-6]
    candidates += [(a + b) / 2 for a, b in zip(all_scores, all_scores[1:])]
    candidates += [all_scores[-1] + 1e-6]

    best = None
    for t in candidates:
        tpr = sum(1 for s in positive_scores if s >= t) / len(positive_scores)
        tnr = sum(1 for s in negative_scores if s < t) / len(negative_scores)
        bal = (tpr + tnr) / 2
        key = (bal, t)  # empate -> limiar maior
        if best is None or key > best[0]:
            best = (key, {"threshold": round(t, 4), "balanced_accuracy": round(bal, 4),
                          "tpr": round(tpr, 4), "tnr": round(tnr, 4)})
    result = best[1]
    result["n_answerable"] = len(positive_scores)
    result["n_unanswerable"] = len(negative_scores)
    return result


# ----------------------------------------------------------------------
# Geração
# ----------------------------------------------------------------------

def is_abstention(answer: str) -> bool:
    """Heurística: a resposta declara que não há informação suficiente?"""
    norm = normalize(answer)
    return any(p in norm for p in ABSTENTION_PATTERNS)


def has_ai_warning(answer: str) -> bool:
    """A resposta traz o aviso de conteúdo gerado por IA?"""
    return "inteligencia artificial" in normalize(answer)


def has_sources_section(answer: str) -> bool:
    """A resposta traz a seção 'Fontes'?"""
    return "fontes" in normalize(answer)


def cites_url(answer: str, urls: Sequence[str]) -> bool:
    """A resposta menciona pelo menos uma das URLs informadas?"""
    return any(u and u in answer for u in urls)
