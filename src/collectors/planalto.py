"""
src/collectors/planalto.py

Coletor para o Portal da Legislação — Presidência da República.
URL base: https://www4.planalto.gov.br/legislacao

AVISO: Os seletores HTML neste arquivo são baseados na estrutura
do site observada em setembro de 2026. A estrutura de sites
governamentais pode mudar. Antes de usar, verifique manualmente
os elementos das páginas listadas nos comentários.

Como adaptar para JavaScript:
Se a página não renderizar os links via requests, use Selenium:
    from selenium import webdriver
    driver = webdriver.Chrome()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
"""

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.collectors.utils import (
    DuplicateFilter,
    HttpClient,
    generate_document_id,
    is_pdf_url,
    now_iso,
    save_metadata,
)

logger = logging.getLogger(__name__)

# URL base do portal do Planalto
BASE_URL = "https://www4.planalto.gov.br"
LEGISLATION_URL = "https://www4.planalto.gov.br/legislacao"

# Padrões de URL que identificam documentos de legislação do Planalto.
# Estes padrões foram identificados inspecionando os links da página.
# Documentos estão sob /ccivil_03/ em diversas subcategorias:
#   /ccivil_03/_ato2023-2026/   → leis e decretos recentes
#   /ccivil_03/decreto/         → decretos
#   /ccivil_03/leis/            → leis antigas
LEGISLATION_PATH_PATTERNS = [
    r"/ccivil_03/",
    r"/legisla\.texto",
]


def is_legislation_link(href: str) -> bool:
    """
    Filtra links que provavelmente levam a documentos de legislação.
    ATENÇÃO: esta heurística precisa ser validada manualmente na página.
    """
    if not href:
        return False
    for pattern in LEGISLATION_PATH_PATTERNS:
        if re.search(pattern, href, re.IGNORECASE):
            return True
    return False


def extract_metadata_from_page(soup: BeautifulSoup, url: str) -> dict:
    """
    Extrai metadados de uma página HTML de legislação do Planalto.

    Elementos a inspecionar na página:
    - Título: <title> ou <h1> com classe 'titulo-norma' ou 'titulo'
    - Data: elemento com texto no formato 'de X de mês de AAAA'
    - Tipo: inferido do início do título (Lei, Decreto, Medida Provisória...)

    ATENÇÃO: os seletores abaixo são sugestões baseadas em padrões comuns
    de sites governamentais. Inspecione o HTML real antes de confiar neles.
    """
    metadata = {
        "title": None,
        "document_type": None,
        "organization": "Presidência da República",
        "publication_date": None,
        "source_url": url,
    }

    # Tentativa 1: <h1> com classe comum em sites do governo
    h1 = soup.find("h1")
    if h1:
        metadata["title"] = h1.get_text(strip=True)

    # Fallback: <title> da página
    if not metadata["title"]:
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

    # Inferir tipo de documento pelo título
    if metadata["title"]:
        title_lower = metadata["title"].lower()
        if title_lower.startswith("lei ") or "lei n" in title_lower[:20]:
            metadata["document_type"] = "lei"
        elif "decreto-lei" in title_lower[:30]:
            metadata["document_type"] = "decreto-lei"
        elif title_lower.startswith("decreto"):
            metadata["document_type"] = "decreto"
        elif "medida provisória" in title_lower[:30]:
            metadata["document_type"] = "medida_provisoria"
        elif "emenda constitucional" in title_lower[:30]:
            metadata["document_type"] = "emenda_constitucional"
        else:
            metadata["document_type"] = "ato_normativo"

    # Buscar data no texto da página com regex
    page_text = soup.get_text()
    date_match = re.search(
        r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})",
        page_text,
        re.IGNORECASE,
    )
    if date_match:
        metadata["publication_date"] = date_match.group(0)

    return metadata


class PlanaltoCollector:
    """
    Coletor para o Portal da Legislação da Presidência da República.

    Funcionamento:
    1. Acessa a página de listagem (start_url).
    2. Extrai links para documentos de legislação.
    3. Para cada link: verifica duplicata, identifica PDF ou HTML.
    4. Baixa o conteúdo e salva localmente.
    5. Extrai metadados e salva em JSONL.
    """

    def __init__(
        self,
        data_dir: Path,
        rate_limit_seconds: float = 1.5,
        timeout: int = 30,
    ):
        self.data_dir = data_dir
        self.pdf_dir = data_dir / "raw" / "pdf"
        self.html_dir = data_dir / "raw" / "html"
        self.metadata_file = data_dir / "processed" / "documents.jsonl"
        self.state_file = data_dir / "processed" / "visited_urls.json"

        # Cria diretórios necessários
        for d in [self.pdf_dir, self.html_dir, self.metadata_file.parent]:
            d.mkdir(parents=True, exist_ok=True)

        self.client = HttpClient(
            rate_limit_seconds=rate_limit_seconds,
            timeout=timeout,
        )
        self.dedup = DuplicateFilter(self.state_file)

    def discover_links(self, page_url: str) -> list[str]:
        """
        Acessa uma página de listagem e retorna URLs de documentos encontrados.

        ATENÇÃO: Esta função busca por links com href que correspondam
        aos padrões de legislação definidos em LEGISLATION_PATH_PATTERNS.
        Se o site mudar, esses padrões precisam ser atualizados.
        """
        logger.info("Descobrindo links em: %s", page_url)
        response = self.client.get(page_url)
        if response is None:
            logger.error("Não foi possível acessar: %s", page_url)
            return []

        soup = BeautifulSoup(response.content, "lxml")
        links = []

        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            # Transforma em URL absoluta
            absolute_url = urljoin(page_url, href)
            if is_legislation_link(absolute_url):
                links.append(absolute_url)

        # Remove duplicatas preservando ordem
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        logger.info("Encontrados %d links únicos em %s", len(unique_links), page_url)
        return unique_links

    def download_pdf(self, url: str, document_id: str) -> Optional[Path]:
        """Baixa um PDF e salva em data/raw/pdf/."""
        dest = self.pdf_dir / f"{document_id}.pdf"
        if dest.exists():
            logger.debug("PDF já existe: %s", dest)
            return dest

        logger.info("Baixando PDF: %s", url)
        response = self.client.get(url, stream=True)
        if response is None:
            return None

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("PDF salvo: %s", dest)
        return dest

    def download_html(self, url: str, document_id: str) -> Optional[Path]:
        """Baixa uma página HTML e salva em data/raw/html/."""
        dest = self.html_dir / f"{document_id}.html"
        if dest.exists():
            logger.debug("HTML já existe: %s", dest)
            return dest

        logger.info("Baixando HTML: %s", url)
        response = self.client.get(url)
        if response is None:
            return None

        with open(dest, "w", encoding="utf-8", errors="replace") as f:
            f.write(response.text)

        logger.info("HTML salvo: %s", dest)
        return dest

    def process_document(self, url: str) -> Optional[dict]:
        """
        Processa um único documento: identifica tipo, baixa e extrai metadados.
        Retorna o dicionário de metadados ou None em caso de falha.
        """
        if self.dedup.is_visited(url):
            logger.debug("URL já visitada, pulando: %s", url)
            return None

        document_id = generate_document_id(url)
        collected_at = now_iso()

        # Verifica Content-Type via HEAD para identificar PDF sem baixar
        head_response = self.client.head(url)
        pdf = is_pdf_url(url, head_response)

        if pdf:
            file_path = self.download_pdf(url, document_id)
            if file_path is None:
                return None
            metadata = {
                "document_id": document_id,
                "title": None,  # será extraído pelo módulo de extração
                "document_type": None,  # idem
                "organization": "Presidência da República",
                "publication_date": None,
                "source_url": url,
                "collected_at": collected_at,
                "file_path": str(file_path.relative_to(self.data_dir.parent)),
                "content_type": "pdf",
            }
        else:
            response = self.client.get(url)
            if response is None:
                return None

            soup = BeautifulSoup(response.content, "lxml")
            metadata = extract_metadata_from_page(soup, url)
            metadata["document_id"] = document_id
            metadata["collected_at"] = collected_at
            metadata["content_type"] = "html"

            file_path = self.html_dir / f"{document_id}.html"
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(response.text)
            metadata["file_path"] = str(file_path.relative_to(self.data_dir.parent))

        # Salva metadados e marca como visitado
        save_metadata(metadata, self.metadata_file)
        self.dedup.mark(url)
        logger.info("Documento processado: %s — %s", document_id, url)
        return metadata

    def collect(
        self,
        start_url: str = LEGISLATION_URL,
        limit: int = 50,
    ) -> list[dict]:
        """
        Ponto de entrada principal do coletor.
        Descobre links a partir de start_url e processa até `limit` documentos.
        """
        logger.info("Iniciando coleta. Limite: %d documentos.", limit)
        links = self.discover_links(start_url)

        if not links:
            logger.warning("Nenhum link encontrado em %s", start_url)
            return []

        collected = []
        for url in links:
            if len(collected) >= limit:
                logger.info("Limite de %d documentos atingido.", limit)
                break

            result = self.process_document(url)
            if result:
                collected.append(result)

        logger.info("Coleta concluída. %d documentos novos coletados.", len(collected))
        return collected
