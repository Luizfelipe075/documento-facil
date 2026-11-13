"""
src/extraction/pdf.py

Extração de texto de PDFs usando PyMuPDF (fitz).
Mantém rastreabilidade por página.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF não instalado. Execute: pip install PyMuPDF")

logger = logging.getLogger(__name__)

# Número mínimo de caracteres por página para considerar que o PDF tem texto.
# PDFs digitalizados (imagens) tipicamente têm 0 ou pouquíssimos caracteres.
MIN_CHARS_PER_PAGE = 50


@dataclass
class PageText:
    """Texto extraído de uma página de PDF com metadados."""
    document_id: str
    page_number: int  # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class PdfExtractionResult:
    """Resultado completo da extração de um PDF."""
    document_id: str
    file_path: str
    pages: list[PageText]
    total_pages: int
    has_text: bool          # False se PDF digitalizado (sem camada de texto)
    full_text: str = field(init=False)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.full_text = "\n\n".join(p.text for p in self.pages if p.text.strip())


def extract_pdf(file_path: Path, document_id: str) -> Optional[PdfExtractionResult]:
    """
    Extrai texto de um arquivo PDF, página por página.

    Args:
        file_path: Caminho para o arquivo PDF.
        document_id: Identificador do documento para rastreabilidade.

    Returns:
        PdfExtractionResult com páginas e texto completo,
        ou None se o arquivo não puder ser aberto.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("Arquivo não encontrado: %s", file_path)
        return None

    try:
        doc = fitz.open(str(file_path))
    except fitz.FileDataError as e:
        logger.error("PDF corrompido ou inválido: %s — %s", file_path, e)
        return None
    except Exception as e:
        logger.error("Erro ao abrir PDF %s: %s", file_path, e)
        return None

    pages: list[PageText] = []
    warnings: list[str] = []
    low_text_pages = 0

    logger.info("Extraindo texto de %s (%d páginas)", file_path.name, doc.page_count)

    for page_index in range(doc.page_count):
        try:
            page = doc[page_index]
            # Extrai texto com layout preservado (melhor para documentos jurídicos)
            text = page.get_text("text")  # 'text' = texto puro com quebras de linha
            page_number = page_index + 1  # converte para 1-indexed

            page_text = PageText(
                document_id=document_id,
                page_number=page_number,
                text=text,
            )
            pages.append(page_text)

            if page_text.char_count < MIN_CHARS_PER_PAGE:
                low_text_pages += 1

        except Exception as e:
            msg = f"Erro na página {page_index + 1}: {e}"
            logger.warning("%s — %s", file_path.name, msg)
            warnings.append(msg)

    total_pages = doc.page_count  # captura antes de fechar o documento
    doc.close()

    # Detecta PDF sem camada de texto (provavelmente digitalizado)
    has_text = low_text_pages < (total_pages * 0.7)
    if not has_text:
        msg = (
            f"PDF possivelmente digitalizado (sem texto selecionável): "
            f"{low_text_pages}/{total_pages} páginas com menos de "
            f"{MIN_CHARS_PER_PAGE} caracteres. Considere OCR."
        )
        logger.warning("%s", msg)
        warnings.append(msg)

    result = PdfExtractionResult(
        document_id=document_id,
        file_path=str(file_path),
        pages=pages,
        total_pages=total_pages,
        has_text=has_text,
        warnings=warnings,
    )

    logger.info(
        "PDF extraído: %s | Páginas: %d | Caracteres: %d | Tem texto: %s",
        document_id,
        result.total_pages,
        len(result.full_text),
        result.has_text,
    )
    return result
