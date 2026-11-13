"""
tests/test_retrieval.py

Testes unitarios para o indice vetorial e busca semantica.
Usa embeddings sinteticos (numpy) para evitar carregar modelos pesados.

NOTA WINDOWS: ChromaDB mantem arquivos SQLite/binarios bloqueados enquanto
o cliente esta ativo. Por isso cada teste fecha explicitamente o cliente
antes do cleanup do diretorio temporario.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def make_fake_embedder(dim: int = 16):
    """Cria um embedder falso que gera vetores aleatorios normalizados."""
    embedder = MagicMock()
    embedder.embedding_dim = dim
    embedder.model_name = "fake-model"

    def fake_embed_texts(texts, **kwargs):
        vecs = np.random.randn(len(texts), dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def fake_embed_query(query, **kwargs):
        vec = np.random.randn(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    embedder.embed_texts.side_effect = fake_embed_texts
    embedder.embed_query.side_effect = fake_embed_query
    return embedder


def make_sample_chunks(n: int = 5, doc_id: str = "doc_teste") -> list[dict]:
    """Gera chunks de amostra para testes."""
    return [
        {
            "chunk_id": f"{doc_id}_chunk_{i:04d}",
            "document_id": doc_id,
            "text": f"Texto do trecho {i} do documento de teste.",
            "chunk_index": i,
            "source_url": f"http://exemplo.gov.br/{doc_id}",
            "title": "Documento de Teste",
            "organization": "Orgao Teste",
            "publication_date": "2024-01-01",
            "page_numbers": [i],
            "token_count": 10,
        }
        for i in range(n)
    ]


def make_index(tmpdir: Path, embedder):
    """Cria e retorna um VectorIndex. Deve ser fechado antes do cleanup do tmpdir."""
    from src.retrieval.index import VectorIndex
    return VectorIndex(persist_dir=tmpdir, embedder=embedder)


def close_index(index) -> None:
    """
    Fecha o cliente ChromaDB para liberar locks de arquivo no Windows.
    ChromaDB nao expoe um metodo close() publico, mas podemos
    deletar o cliente interno para acionar o GC.
    """
    try:
        if hasattr(index, '_client') and index._client is not None:
            # Tenta fechar a conexao SQLite internamente
            if hasattr(index._client, '_system'):
                index._client._system.stop()
            del index._client
            index._client = None
        if hasattr(index, '_collection'):
            index._collection = None
    except Exception:
        pass


class TestVectorIndex:
    """Testa o indice vetorial ChromaDB."""

    def test_index_cria_e_persiste(self):
        """Testa que o indice e criado sem erros."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            assert index.count == 0
        finally:
            close_index(index)

    def test_index_build_insere_chunks(self):
        """Testa que os chunks sao inseridos corretamente."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=5)
            index.build(chunks, embedder=embedder)
            assert index.count == 5
        finally:
            if index:
                close_index(index)

    def test_index_search_retorna_resultados(self):
        """Testa que a busca retorna a quantidade correta de resultados."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=10)
            index.build(chunks, embedder=embedder)

            query_vec = embedder.embed_query("pergunta de teste")
            results = index.search(query_vec, k=3)
            assert len(results) == 3
        finally:
            if index:
                close_index(index)

    def test_search_resultado_tem_campos_obrigatorios(self):
        """Testa que cada resultado contem os campos necessarios para rastreabilidade."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=5)
            index.build(chunks, embedder=embedder)

            query_vec = embedder.embed_query("pergunta")
            results = index.search(query_vec, k=1)

            assert len(results) == 1
            r = results[0]
            required_fields = ["chunk_id", "score", "text", "document_id", "source_url"]
            for field in required_fields:
                assert field in r, f"Campo '{field}' ausente no resultado"
        finally:
            if index:
                close_index(index)

    def test_search_score_entre_zero_e_um(self):
        """Testa que os scores de similaridade estao no intervalo [0, 1]."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=5)
            index.build(chunks, embedder=embedder)

            query_vec = embedder.embed_query("pergunta")
            results = index.search(query_vec, k=5)

            for r in results:
                assert 0.0 <= r["score"] <= 1.0, f"Score fora do intervalo: {r['score']}"
        finally:
            if index:
                close_index(index)

    def test_upsert_nao_duplica(self):
        """Testa que inserir os mesmos chunks duas vezes nao duplica."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=5)
            index.build(chunks, embedder=embedder)
            index.build(chunks, embedder=embedder)  # segunda vez

            assert index.count == 5  # nao deve duplicar (upsert)
        finally:
            if index:
                close_index(index)

    def test_indice_vazio_retorna_lista_vazia(self):
        """Testa que busca em indice vazio retorna lista vazia com aviso."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)

            query_vec = embedder.embed_query("pergunta")
            results = index.search(query_vec, k=5)
            assert results == []
        finally:
            if index:
                close_index(index)

    def test_k_maior_que_total_nao_falha(self):
        """Testa que pedir mais resultados que o total disponivel nao causa erro."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb nao instalado")

        tmpdir = Path(tempfile.mkdtemp())
        index = None
        try:
            embedder = make_fake_embedder()
            index = make_index(tmpdir, embedder)
            chunks = make_sample_chunks(n=3)
            index.build(chunks, embedder=embedder)

            query_vec = embedder.embed_query("pergunta")
            results = index.search(query_vec, k=10)  # pede 10, tem 3
            assert len(results) == 3
        finally:
            if index:
                close_index(index)
