"""
src/evaluation/retrieval.py

Avaliação da qualidade da recuperação de chunks.
Métricas: Recall@1, Recall@3, Recall@5

Recall@K = (perguntas em que o documento correto aparece no Top K) / (total de perguntas)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_retrieval(
    questions_file: Path,
    vectorstore_dir: Optional[Path] = None,
    k_values: list[int] = [1, 3, 5],
) -> dict:
    """
    Avalia a recuperação usando um conjunto de perguntas com respostas esperadas.

    Args:
        questions_file: Caminho para test_questions.json
        vectorstore_dir: Diretório do ChromaDB.
        k_values: Valores de K para calcular Recall@K.

    Returns:
        Dicionário com métricas e detalhes por pergunta.
    """
    from src.retrieval.search import search

    if vectorstore_dir is None:
        vectorstore_dir = Path("vectorstore")

    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    max_k = max(k_values)
    results = []

    for q in questions:
        question = q["question"]
        expected_docs = set(q.get("expected_documents", []))

        if not expected_docs:
            continue

        chunks = search(question, k=max_k, vectorstore_dir=vectorstore_dir)
        retrieved_docs = [c["document_id"] for c in chunks]

        row = {
            "question": question,
            "expected_documents": list(expected_docs),
            "retrieved_documents": retrieved_docs,
        }

        for k in k_values:
            top_k_docs = set(retrieved_docs[:k])
            hit = bool(expected_docs & top_k_docs)
            row[f"recall_at_{k}"] = hit

        results.append(row)

    # Calcula métricas agregadas
    metrics = {}
    for k in k_values:
        key = f"recall_at_{k}"
        hits = sum(1 for r in results if r.get(key, False))
        total = len(results)
        recall = hits / total if total > 0 else 0.0
        metrics[f"Recall@{k}"] = round(recall, 4)
        logger.info("Recall@%d: %.2f%% (%d/%d)", k, recall * 100, hits, total)

    return {
        "metrics": metrics,
        "total_questions": len(results),
        "details": results,
    }
