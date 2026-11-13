"""
src/collectors/utils.py

Utilitários compartilhados para todos os coletores.
Contém: HttpClient, RateLimiter, DuplicateFilter, funções de metadados.
"""

import json
import logging
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# User-Agent realista para evitar bloqueios simples.
# Identifica o projeto para transparência com os servidores.
DEFAULT_USER_AGENT = (
    "DocumentoFacil/1.0 (projeto acadêmico; coleta de documentos públicos; "
    "https://github.com/zemei/documento-facil)"
)


class RateLimiter:
    """
    Controla o intervalo mínimo entre requisições para um mesmo domínio.
    Evita sobrecarregar servidores públicos.
    """

    def __init__(self, seconds: float = 1.5):
        self.seconds = seconds
        self._last_request: dict[str, float] = {}

    def wait(self, domain: str = "default") -> None:
        """Aguarda o tempo necessário antes de uma nova requisição."""
        now = time.monotonic()
        last = self._last_request.get(domain, 0.0)
        elapsed = now - last
        if elapsed < self.seconds:
            sleep_time = self.seconds - elapsed
            logger.debug("RateLimiter: aguardando %.2fs para %s", sleep_time, domain)
            time.sleep(sleep_time)
        self._last_request[domain] = time.monotonic()


class DuplicateFilter:
    """
    Mantém um conjunto de URLs já visitadas para evitar downloads duplicados.
    Persiste o conjunto em um arquivo JSON entre execuções.
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._visited: set[str] = self._load()

    def _load(self) -> set[str]:
        """Carrega URLs visitadas do arquivo de estado, se existir."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("DuplicateFilter: %d URLs já visitadas carregadas.", len(data))
                return set(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("DuplicateFilter: erro ao carregar estado: %s", e)
        return set()

    def save(self) -> None:
        """Persiste o conjunto de URLs visitadas no arquivo JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(sorted(self._visited), f, ensure_ascii=False, indent=2)

    def is_visited(self, url: str) -> bool:
        """Retorna True se a URL já foi processada."""
        return url in self._visited

    def mark(self, url: str) -> None:
        """Marca uma URL como visitada e salva o estado."""
        self._visited.add(url)
        self.save()

    def __len__(self) -> int:
        return len(self._visited)


class HttpClient:
    """
    Cliente HTTP com retry automático, timeout configurável e User-Agent.
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_seconds: float = 1.5,
    ):
        self.timeout = timeout
        self.rate_limiter = RateLimiter(seconds=rate_limit_seconds)
        self.session = self._build_session(user_agent, max_retries)

    def _build_session(self, user_agent: str, max_retries: int) -> requests.Session:
        """Cria uma sessão requests com retry automático em erros transitórios."""
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})

        # Retry em status de servidor ou erros de rede
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=2,  # espera 2s, 4s, 8s entre tentativas
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Faz GET com rate limiting e tratamento de exceções.
        Retorna None em caso de erro.
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        self.rate_limiter.wait(domain)

        try:
            response = self.session.get(url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            logger.error("Timeout ao acessar: %s", url)
        except requests.exceptions.HTTPError as e:
            logger.error("Erro HTTP %s: %s", e.response.status_code, url)
        except requests.exceptions.ConnectionError:
            logger.error("Erro de conexão: %s", url)
        except requests.exceptions.RequestException as e:
            logger.error("Erro inesperado em %s: %s", url, e)
        return None

    def head(self, url: str) -> Optional[requests.Response]:
        """HEAD request para verificar Content-Type sem baixar o corpo."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        self.rate_limiter.wait(domain)
        try:
            return self.session.head(url, timeout=self.timeout, allow_redirects=True)
        except requests.exceptions.RequestException as e:
            logger.debug("HEAD falhou para %s: %s", url, e)
        return None


def is_pdf_url(url: str, response: Optional[requests.Response] = None) -> bool:
    """
    Detecta se uma URL aponta para um PDF.
    Verifica a extensão da URL e o Content-Type da resposta.
    """
    url_lower = url.lower().split("?")[0]  # ignora query string
    if url_lower.endswith(".pdf"):
        return True
    if response is not None:
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" in content_type:
            return True
    return False


def generate_document_id(url: str) -> str:
    """
    Gera um document_id estável e legível a partir da URL.
    Usa os últimos segmentos do path + hash curto para unicidade.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    slug = "_".join(path_parts[-3:]) if path_parts else "doc"
    # Remove caracteres problemáticos
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in slug).lower()
    short_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{slug}_{short_hash}"


def save_metadata(record: dict, jsonl_path: Path) -> None:
    """
    Adiciona um registro de metadados ao arquivo JSONL.
    Cria o arquivo se não existir.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def now_iso() -> str:
    """Retorna o timestamp atual em formato ISO 8601 com fuso horário."""
    return datetime.now(timezone.utc).isoformat()
