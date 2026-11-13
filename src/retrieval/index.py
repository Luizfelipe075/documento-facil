"""
src/retrieval/index.py

Índice vetorial usando ChromaDB para armazenamento e busca semântica.

Vantagens do ChromaDB sobre FAISS para este projeto:
- Persiste vetores e metadados juntos (sem gerenciar arquivo separado)
- API simples e pythônica
- Suporta filtros por metadados
- Adequado para dezenas de milhares de chunks

Para escalar para milhões de chunks, considere migrar para FAISS ou Pinecone.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "documento_facil"


class VectorIndex:
    """
    Índice vetorial com ChromaDB para busca semântica de chunks.

    Metadados armazenados por chunk:
    - document_id, title, source_url, organization,
      publication_date, chunk_index, token_count
    """

    def __init__(self, persist_dir: Path, embedder=None):
        """
        Args:
            persist_dir: Diretório onde o ChromaDB armazenará os dados.
            embedder: Instância de Embedder (opcional; se None, usa funções externas).
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self) -> None:
        """Inicializa o cliente ChromaDB com persistência."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},  # similaridade cosseno
            )
            logger.info(
                "ChromaDB inicializado em %s | Coleção: %s | Itens: %d",
                self.persist_dir,
                COLLECTION_NAME,
                self._collection.count(),
            )
        except ImportError:
            raise ImportError("chromadb não instalado. Execute: pip install chromadb")

    @property
    def count(self) -> int:
        """Número de chunks armazenados no índice."""
        return self._collection.count()

    def build(self, chunks: list[dict], embedder=None) -> None:
        """
        Gera embeddings e insere chunks no índice ChromaDB.

        Args:
            chunks: Lista de dicionários de chunks (output do chunker).
            embedder: Instância de Embedder. Usa self.embedder se não fornecido.
        """
        emb = embedder or self.embedder
        if emb is None:
            raise ValueError("Embedder não fornecido.")

        if not chunks:
            logger.warning("Nenhum chunk para indexar.")
            return

        logger.info("Indexando %d chunks...", len(chunks))

        # Processa em batches para evitar uso excessivo de memória
        BATCH_SIZE = 100
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i: i + BATCH_SIZE]
            self._insert_batch(batch, emb)
            logger.info("Batch %d/%d inserido", i // BATCH_SIZE + 1, (len(chunks) - 1) // BATCH_SIZE + 1)

        logger.info("Indexação concluída. Total no índice: %d", self.count)

    def _insert_batch(self, batch: list[dict], embedder) -> None:
        """Insere um batch de chunks no ChromaDB."""
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]

        # Gera embeddings para o batch
        embeddings = embedder.embed_texts(texts)

        # Prepara metadados (ChromaDB não aceita None; converte para string vazia)
        metadatas = []
        for c in batch:
            meta = {
                "document_id": c.get("document_id", ""),
                "title": c.get("title") or "",
                "source_url": c.get("source_url", ""),
                "organization": c.get("organization") or "",
                "publication_date": c.get("publication_date") or "",
                "chunk_index": c.get("chunk_index", 0),
                "token_count": c.get("token_count", 0),
                "page_numbers": json.dumps(c.get("page_numbers", [])),
            }
            metadatas.append(meta)

        # Upsert: insere ou atualiza se o chunk_id já existir
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query_embedding, k: int = 5) -> list[dict]:
        """
        Busca os k chunks mais similares à query.

        Args:
            query_embedding: Vetor da query (numpy array).
            k: Número de resultados a retornar.

        Returns:
            Lista de resultados com score, texto e metadados.
        """
        if self.count == 0:
            logger.warning("Índice vazio. Execute build() primeiro.")
            return []

        k = min(k, self.count)  # não pedir mais que o disponível

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
            # ChromaDB com cosine retorna distâncias (0=idêntico, 2=oposto)
            # Converte para score de similaridade [0, 1]
            score = 1.0 - (dist / 2.0)
            hit = {
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": doc,
                "document_id": meta.get("document_id", ""),
                "title": meta.get("title", ""),
                "source_url": meta.get("source_url", ""),
                "organization": meta.get("organization", ""),
                "publication_date": meta.get("publication_date", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "page_numbers": json.loads(meta.get("page_numbers", "[]")),
            }
            hits.append(hit)

        return hits

    def reset(self) -> None:
        """Remove todos os chunks do índice. Use com cuidado."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Índice resetado. Todos os dados foram removidos.")
