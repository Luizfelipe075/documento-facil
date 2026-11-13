"""
src/extraction/html.py

Extração de texto limpo de páginas HTML.
Remove navegação, menus, rodapés e outros elementos que não fazem parte
do conteúdo principal do documento.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# Tags HTML que contêm navegação e elementos de interface, não conteúdo.
# Esses elementos são removidos antes da extração do texto.
NAV_TAGS = ["nav", "header", "footer", "aside"]

# Classes e IDs CSS comuns em sites governamentais que indicam
# elementos de navegação ou de interface (não conteúdo).
# ATENÇÃO: estes seletores são heurísticos. Inspecione o HTML real para ajustá-los.
NAV_CLASSES = [
    "menu", "navbar", "navigation", "breadcrumb", "sidebar",
    "footer", "header", "rodape", "cabecalho", "topo",
    "btn", "button", "social", "share", "cookie",
]


@dataclass
class HtmlExtractionResult:
    """Resultado da extração de texto de uma página HTML."""
    document_id: str
    source_url: str
    title: Optional[str]
    body: str
    date: Optional[str]
    organization: Optional[str]
    document_type: Optional[str]
    char_count: int


def _remove_noise_elements(soup: BeautifulSoup) -> None:
    """
    Remove elementos de interface (não conteúdo) do HTML.
    Modifica o objeto soup in-place.
    """
    # Remove tags de navegação padrão
    for tag_name in NAV_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove tags de script e estilo
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # Remove elementos por classe/id de navegação
    for tag in soup.find_all(True):
        tag_classes = " ".join(tag.get("class", [])).lower()
        tag_id = (tag.get("id") or "").lower()
        for nav_class in NAV_CLASSES:
            if nav_class in tag_classes or nav_class in tag_id:
                tag.decompose()
                break


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    """Extrai o título do documento HTML."""
    # Prioridade: h1 > h2 > title
    for tag_name in ["h1", "h2"]:
        tag = soup.find(tag_name)
        if tag:
            return tag.get_text(strip=True)
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    return None


def _extract_date(soup: BeautifulSoup, text: str) -> Optional[str]:
    """Extrai data de publicação do HTML ou do texto da página."""
    # Busca meta tags de data
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if "date" in name or "publicacao" in name:
            return meta.get("content")

    # Busca padrão de data no texto
    date_match = re.search(
        r"(\d{1,2})\s+de\s+\w+\s+de\s+(\d{4})",
        text,
        re.IGNORECASE,
    )
    if date_match:
        return date_match.group(0)

    # Busca formato DD/MM/AAAA
    date_match2 = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    if date_match2:
        return date_match2.group(1)

    return None


def _infer_document_type(title: Optional[str]) -> Optional[str]:
    """Infere o tipo de documento a partir do título."""
    if not title:
        return None
    t = title.lower()
    if "lei complementar" in t:
        return "lei_complementar"
    elif t.startswith("lei ") or "lei n" in t[:20]:
        return "lei"
    elif "decreto-lei" in t[:30]:
        return "decreto-lei"
    elif t.startswith("decreto"):
        return "decreto"
    elif "medida provisória" in t[:30]:
        return "medida_provisoria"
    elif "emenda constitucional" in t[:30]:
        return "emenda_constitucional"
    elif "constituição" in t[:30]:
        return "constituicao"
    elif "portaria" in t[:20]:
        return "portaria"
    elif "resolução" in t[:20]:
        return "resolucao"
    return "ato_normativo"


def extract_html(
    file_path: Path,
    document_id: str,
    source_url: str,
    organization: str = "Presidência da República",
) -> Optional[HtmlExtractionResult]:
    """
    Extrai texto de um arquivo HTML salvo localmente.

    Args:
        file_path: Caminho para o arquivo HTML.
        document_id: Identificador do documento.
        source_url: URL original do documento.
        organization: Órgão responsável pelo documento.

    Returns:
        HtmlExtractionResult ou None em caso de erro.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("Arquivo não encontrado: %s", file_path)
        return None

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            html_content = f.read()
    except IOError as e:
        logger.error("Erro ao ler arquivo HTML %s: %s", file_path, e)
        return None

    soup = BeautifulSoup(html_content, "lxml")

    # Remove elementos de interface antes de extrair texto
    _remove_noise_elements(soup)

    title = _extract_title(soup)

    # Extrai o corpo principal
    # Tenta encontrar o elemento principal de conteúdo
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="conteudo")
        or soup.find(id="content")
        or soup.find(id="main")
        or soup.find("body")
    )

    if main_content:
        body = main_content.get_text(separator="\n", strip=True)
    else:
        body = soup.get_text(separator="\n", strip=True)

    date = _extract_date(soup, body)
    document_type = _infer_document_type(title)

    result = HtmlExtractionResult(
        document_id=document_id,
        source_url=source_url,
        title=title,
        body=body,
        date=date,
        organization=organization,
        document_type=document_type,
        char_count=len(body),
    )

    logger.info(
        "HTML extraído: %s | Título: %s | Caracteres: %d",
        document_id,
        title,
        result.char_count,
    )
    return result
