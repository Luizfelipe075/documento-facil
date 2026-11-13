"""
tests/test_extraction.py

Testes unitarios para os modulos de extracao de texto.
Usa documentos sinteticos para evitar dependencia de rede.
"""

import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# Testes: extraction/html.py
# ============================================================

class TestHtmlExtraction:
    """Testa a extracao de texto de HTML."""

    def _write_html(self, content: str) -> Path:
        """Cria um arquivo HTML temporario e retorna o caminho."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", mode="w", encoding="utf-8", delete=False
        )
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_extrai_titulo_h1(self):
        from src.extraction.html import extract_html
        html = "<html><body><h1>Lei no 9.394</h1><p>Texto da lei.</p></body></html>"
        path = self._write_html(html)
        result = extract_html(path, "doc_001", "http://exemplo.gov.br")
        assert result is not None
        assert "Lei" in result.title
        path.unlink()

    def test_extrai_corpo_principal(self):
        from src.extraction.html import extract_html
        html = """
        <html><body>
            <nav>Menu de navegacao</nav>
            <main><p>Art. 1o Esta lei estabelece as diretrizes.</p></main>
            <footer>Rodape do site</footer>
        </body></html>
        """
        path = self._write_html(html)
        result = extract_html(path, "doc_002", "http://exemplo.gov.br")
        assert result is not None
        assert "Art. 1" in result.body
        path.unlink()

    def test_remove_nav_e_footer(self):
        from src.extraction.html import extract_html
        html = """
        <html><body>
            <nav>Ir para pagina inicial</nav>
            <main><p>Conteudo importante do documento.</p></main>
            <footer>Copyright 2024</footer>
        </body></html>
        """
        path = self._write_html(html)
        result = extract_html(path, "doc_003", "http://exemplo.gov.br")
        assert result is not None
        assert "Ir para pagina inicial" not in result.body
        assert "Copyright" not in result.body
        path.unlink()

    def test_arquivo_nao_encontrado_retorna_none(self):
        from src.extraction.html import extract_html
        result = extract_html(
            Path("/caminho/que/nao/existe.html"),
            "doc_404",
            "http://exemplo.gov.br",
        )
        assert result is None

    def test_infere_tipo_lei(self):
        from src.extraction.html import extract_html
        html = "<html><body><h1>Lei no 14.133, de 2021</h1><p>Texto.</p></body></html>"
        path = self._write_html(html)
        result = extract_html(path, "doc_005", "http://exemplo.gov.br")
        assert result is not None
        assert result.document_type == "lei"
        path.unlink()

    def test_infere_tipo_decreto(self):
        from src.extraction.html import extract_html
        html = "<html><body><h1>Decreto no 10.000</h1><p>Texto.</p></body></html>"
        path = self._write_html(html)
        result = extract_html(path, "doc_006", "http://exemplo.gov.br")
        assert result is not None
        assert result.document_type == "decreto"
        path.unlink()

    def test_extrai_data_por_regex(self):
        from src.extraction.html import extract_html
        html = """
        <html><body>
            <h1>Lei Teste</h1>
            <p>Publicada em 20 de dezembro de 1996.</p>
        </body></html>
        """
        path = self._write_html(html)
        result = extract_html(path, "doc_007", "http://exemplo.gov.br")
        assert result is not None
        assert result.date is not None
        assert "1996" in result.date
        path.unlink()


# ============================================================
# Testes: extraction/pdf.py
# ============================================================

class TestPdfExtraction:
    """Testa a extracao de texto de PDFs."""

    def test_arquivo_pdf_inexistente_retorna_none(self):
        from src.extraction.pdf import extract_pdf
        result = extract_pdf(Path("/nao/existe.pdf"), "doc_pdf_001")
        assert result is None

    def test_arquivo_nao_pdf_retorna_none(self):
        """Testa que um arquivo corrompido/invalido retorna None."""
        from src.extraction.pdf import extract_pdf
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf", mode="wb", delete=False
        )
        tmp.write(b"isto nao e um PDF valido")
        tmp.close()
        result = extract_pdf(Path(tmp.name), "doc_invalido")
        assert result is None
        # No Windows, PyMuPDF pode manter lock brevemente; ignora erro de cleanup
        try:
            Path(tmp.name).unlink()
        except PermissionError:
            pass  # lock temporario do Windows, nao e falha do codigo

    def test_pdf_valido_extrai_texto(self):
        """
        Testa extracao em PDF real se PyMuPDF estiver instalado.
        Cria um PDF minimo usando apenas bytes validos.
        """
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF nao instalado")

        # Cria um PDF minimo com texto usando PyMuPDF
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.close()
        pdf_path = Path(tmp.name)

        doc = fitz.open()
        page = doc.new_page()
        # Texto com mais de 50 caracteres para que has_text seja True
        legal_text = (
            "Art. 1o Esta lei estabelece as diretrizes e bases da educacao nacional. "
            "Art. 2o A educacao e dever da familia e do Estado."
        )
        page.insert_text((72, 72), legal_text)
        doc.save(str(pdf_path))
        doc.close()

        from src.extraction.pdf import extract_pdf
        result = extract_pdf(pdf_path, "doc_pdf_teste")

        assert result is not None
        assert result.total_pages == 1
        assert "Art." in result.full_text or "teste" in result.full_text
        assert result.has_text is True

        # No Windows, PyMuPDF pode manter lock brevemente
        try:
            pdf_path.unlink()
        except PermissionError:
            pass  # lock temporario do Windows, nao e falha do codigo
