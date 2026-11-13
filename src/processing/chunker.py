"""
src/processing/chunker.py

Divisão de documentos em chunks com sobreposição.
Usa tiktoken para contagem de tokens, garantindo compatibilidade com LLMs.

Estratégia de chunking:
- Divide preferencialmente em parágrafos (quebras duplas de linha).
- Se um parágrafo ultrapassar chunk_size, divide por sentenças.
- Adiciona overlap pegando tokens do final do chunk anterior.
- NUNCA divide no meio de um artigo jurídico se possível.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

try:
    import tiktoken
except ImportError:
    raise ImportError("tiktoken não instalado. Execute: pip install tiktoken")

logger = logging.getLogger(__name__)

# Encoding padrão: cl100k_base (usado pelo GPT-4 e Gemini-compatible)
DEFAULT_ENCODING = "cl100k_base"


class _RegexEncoding:
    """
    Fallback offline quando o vocabulário do tiktoken não pode ser baixado.

    O tiktoken baixa o vocabulário (cl100k_base) da internet na primeira
    execução. Sem rede (ou atrás de proxy), isso falha. Este codificador
    aproxima "tokens" por palavras/pontuação, de forma REVERSÍVEL (decode
    reconstrói o texto exatamente). A contagem difere um pouco da do
    tiktoken, então chunks podem ter tamanho ligeiramente diferente.
    """

    _PIECE_RE = re.compile(r"\s*\w+|\s*[^\w\s]|\s+")

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._pieces: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in self._PIECE_RE.findall(text):
            if piece not in self._ids:
                self._ids[piece] = len(self._pieces)
                self._pieces.append(piece)
            ids.append(self._ids[piece])
        return ids

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._pieces[t] for t in tokens)


_ENCODING_CACHE: dict = {}


def _load_encoding(encoding_name: str):
    """
    Carrega o encoding do tiktoken; usa o fallback offline se falhar.
    O resultado (inclusive a falha) é cacheado: o aviso sai uma vez só e não
    se tenta baixar o vocabulário de novo a cada Chunker criado.
    """
    if encoding_name in _ENCODING_CACHE:
        return _ENCODING_CACHE[encoding_name]
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception as e:  # falha de rede, proxy, cache indisponível...
        logger.warning(
            "Não foi possível carregar o encoding '%s' do tiktoken (%s). "
            "Usando contagem aproximada por palavras (fallback offline).",
            encoding_name, type(e).__name__,
        )
        enc = _RegexEncoding()
    _ENCODING_CACHE[encoding_name] = enc
    return enc


@dataclass
class Chunk:
    """Um trecho de documento com todos os metadados de rastreabilidade."""
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    source_url: str
    title: Optional[str] = None
    organization: Optional[str] = None
    publication_date: Optional[str] = None
    page_numbers: list[int] = field(default_factory=list)  # páginas cobertas
    token_count: int = 0

    def to_dict(self) -> dict:
        """Converte para dicionário para serialização em JSONL."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "source_url": self.source_url,
            "title": self.title,
            "organization": self.organization,
            "publication_date": self.publication_date,
            "page_numbers": self.page_numbers,
            "token_count": self.token_count,
        }


class Chunker:
    """
    Divide texto em chunks de tamanho configurável com sobreposição.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = _load_encoding(encoding_name)

    def _tokenize(self, text: str) -> list[int]:
        """Converte texto em lista de tokens."""
        return self.enc.encode(text)

    def _detokenize(self, tokens: list[int]) -> str:
        """Converte tokens de volta para texto."""
        return self.enc.decode(tokens)

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """
        Divide texto em parágrafos usando quebras duplas de linha.
        Tenta preservar artigos, incisos e alíneas intactos.
        """
        # Quebra em parágrafos (dupla quebra de linha)
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: dict,
    ) -> list[Chunk]:
        """
        Divide um texto em chunks com sobreposição.

        Args:
            text: Texto limpo e normalizado.
            document_id: ID do documento para rastreabilidade.
            metadata: Dicionário com source_url, title, organization, etc.

        Returns:
            Lista de Chunks com todos os metadados preenchidos.
        """
        source_url = metadata.get("source_url", "")
        title = metadata.get("title")
        organization = metadata.get("organization")
        publication_date = metadata.get("publication_date")

        # Tokeniza o texto completo
        all_tokens = self._tokenize(text)
        total_tokens = len(all_tokens)

        if total_tokens == 0:
            logger.warning("Texto vazio para documento: %s", document_id)
            return []

        logger.info(
            "Chunking: %s | %d tokens | chunk_size=%d | overlap=%d",
            document_id, total_tokens, self.chunk_size, self.overlap,
        )

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = all_tokens[start:end]
            chunk_text = self._detokenize(chunk_tokens)

            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                source_url=source_url,
                title=title,
                organization=organization,
                publication_date=publication_date,
                token_count=len(chunk_tokens),
            )
            chunks.append(chunk)

            # Próximo chunk começa com sobreposição
            step = self.chunk_size - self.overlap
            start += step
            chunk_index += 1

        logger.info(
            "Chunking concluído: %s | %d chunks gerados",
            document_id, len(chunks),
        )
        return chunks


def chunk_document(
    text: str,
    document_id: str,
    metadata: dict,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """
    Função de conveniência: cria um Chunker e divide o documento.

    Returns:
        Lista de dicionários prontos para serialização em JSONL.
    """
    chunker = Chunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_text(text, document_id, metadata)
    return [c.to_dict() for c in chunks]
