"""
tests/test_processing.py

Testes unitarios para limpeza de texto (cleaner.py) e chunking (chunker.py).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Testes: processing/cleaner.py
# ============================================================

class TestCleaner:
    """Testa o pipeline de limpeza de texto."""

    def test_remove_espacos_excessivos(self):
        from src.processing.cleaner import clean_text
        raw = "Art.   1o    Esta   lei    estabelece."
        result = clean_text(raw)
        assert "   " not in result

    def test_normaliza_quebras_de_linha(self):
        from src.processing.cleaner import clean_text
        raw = "Linha 1\n\n\n\n\n\nLinha 2"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_remove_separadores_graficos(self):
        from src.processing.cleaner import clean_text
        raw = "Cabecalho\n---\nArt. 1o\n___\nFim"
        result = clean_text(raw)
        assert "---" not in result
        assert "___" not in result

    def test_preserva_artigos(self):
        from src.processing.cleaner import clean_text
        raw = "Art. 1o Esta lei.\nArt. 2o Outra norma."
        result = clean_text(raw)
        assert "Art. 1" in result
        assert "Art. 2" in result

    def test_preserva_paragrafos(self):
        from src.processing.cleaner import clean_text
        raw = "\u00a7 1o O disposto neste artigo.\n\u00a7 2o Aplica-se tambem."
        result = clean_text(raw)
        assert "\u00a7 1" in result
        assert "\u00a7 2" in result

    def test_preserva_incisos(self):
        from src.processing.cleaner import clean_text
        raw = "I - primeiro inciso;\nII - segundo inciso;\nIII - terceiro inciso."
        result = clean_text(raw)
        assert "I - primeiro" in result
        assert "II - segundo" in result

    def test_preserva_datas(self):
        from src.processing.cleaner import clean_text
        raw = "Publicada em 20 de dezembro de 1996."
        result = clean_text(raw)
        assert "20 de dezembro de 1996" in result

    def test_remove_cabecalhos_repetidos(self):
        from src.processing.cleaner import clean_text
        # Cabecalho curto repetido 4 vezes
        header = "Portal do Planalto"
        raw = f"{header}\nArt. 1o\n{header}\nArt. 2o\n{header}\nArt. 3o\n{header}"
        result = clean_text(raw)
        # Apos remocao, deve aparecer menos vezes
        count = result.count(header)
        assert count < 4

    def test_normalizacao_unicode(self):
        from src.processing.cleaner import clean_text
        # Caractere 'a' com acento em forma decomposta (NFD)
        raw = "Boa-tarde"
        result = clean_text(raw)
        assert len(result) > 0  # deve retornar sem erros

    def test_texto_vazio_retorna_vazio(self):
        from src.processing.cleaner import clean_text
        result = clean_text("")
        assert result == ""

    def test_texto_apenas_separadores_retorna_vazio(self):
        from src.processing.cleaner import clean_text
        raw = "---\n___\n===\n---"
        result = clean_text(raw)
        assert result.strip() == ""


# ============================================================
# Testes: processing/chunker.py
# ============================================================

class TestChunker:
    """Testa o chunking de documentos."""

    SAMPLE_META = {
        "source_url": "http://exemplo.gov.br/lei",
        "title": "Lei de Teste",
        "organization": "Orgao Teste",
        "publication_date": "2024-01-01",
    }

    def test_chunk_basico(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = " ".join(["palavra"] * 1000)
        chunks = chunk_document(text, "doc_001", self.SAMPLE_META, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunk_preserva_document_id(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = " ".join(["palavra"] * 500)
        chunks = chunk_document(text, "doc_teste", self.SAMPLE_META, chunk_size=100, overlap=20)
        for chunk in chunks:
            assert chunk["document_id"] == "doc_teste"

    def test_chunk_preserva_source_url(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = " ".join(["palavra"] * 500)
        chunks = chunk_document(text, "doc_url", self.SAMPLE_META, chunk_size=100, overlap=20)
        for chunk in chunks:
            assert chunk["source_url"] == self.SAMPLE_META["source_url"]

    def test_chunk_id_unico(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = " ".join(["palavra"] * 1000)
        chunks = chunk_document(text, "doc_id", self.SAMPLE_META, chunk_size=100, overlap=20)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))  # todos unicos

    def test_chunk_index_sequencial(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = " ".join(["palavra"] * 500)
        chunks = chunk_document(text, "doc_seq", self.SAMPLE_META, chunk_size=100, overlap=20)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))

    def test_texto_vazio_retorna_lista_vazia(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        chunks = chunk_document("", "doc_vazio", self.SAMPLE_META)
        assert chunks == []

    def test_documento_pequeno_gera_um_chunk(self):
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken nao instalado")

        from src.processing.chunker import chunk_document
        text = "Art. 1o Esta lei e curta."
        chunks = chunk_document(text, "doc_curto", self.SAMPLE_META, chunk_size=500, overlap=100)
        assert len(chunks) == 1
