"""
src/retrieval/search.py

Interface de busca semântica de alto nível.
Conecta o embedder ao índice vetorial.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Instâncias singleton carregadas uma vez
_embedder = None
_index = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from src.embeddings.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _get_index(vectorstore_dir: Path):
    global _index
    if _index is None:
        from src.retrieval.index import VectorIndex
        embedder = _get_embedder()
        _index = VectorIndex(persist_dir=vectorstore_dir, embedder=embedder)
    return _index


def search(
    query: str,
    k: int = 5,
    vectorstore_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Busca os k trechos mais relevantes para uma query.

    Args:
        query: Pergunta do usuário em linguagem natural.
        k: Número de resultados a retornar.
        vectorstore_dir: Diretório do ChromaDB. Padrão: ./vectorstore

    Returns:
        Lista de dicionários com:
        {
            'chunk_id': str,
            'score': float,       # similaridade cosseno [0, 1]
            'text': str,
            'document_id': str,
            'title': str,
            'source_url': str,
            'organization': str,
            'publication_date': str,
            'chunk_index': int,
            'page_numbers': list[int],
        }
    """
    if vectorstore_dir is None:
        vectorstore_dir = Path("vectorstore")

    logger.info("Buscando: %r | k=%d", query, k)

    embedder = _get_embedder()
    index = _get_index(vectorstore_dir)

    query_embedding = embedder.embed_query(query)
    results = index.search(query_embedding, k=k)

    logger.info("Encontrados %d resultados para: %r", len(results), query)
    return results


def reset_singletons() -> None:
    """Reseta os singletons (útil em testes)."""
    global _embedder, _index
    _embedder = None
    _index = None
