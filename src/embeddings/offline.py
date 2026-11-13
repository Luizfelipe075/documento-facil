"""
src/embeddings/offline.py

Embedder lexical offline (hashing trick) com a MESMA interface do Embedder.

Para que serve:
- Rodar testes e experimentos sem baixar o modelo multilingual-e5-large
  (~2 GB) e sem GPU/internet.
- Servir de LINHA DE BASE nos experimentos: se o e5 não superar
  um baseline lexical, algo está errado na configuração.

Para que NÃO serve:
- Não entende sinônimos nem semântica. Os resultados com este embedder
  NÃO substituem os do modelo real e não devem ir para o relatório técnico
  como se fossem equivalentes.

Determinístico: usa zlib.crc32 (o hash() do Python é aleatorizado por processo).
"""

import re
import unicodedata
import zlib

import numpy as np

# Stopwords mínimas do português — sem isso, "qual", "de", "o" dominam o vetor.
_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das",
    "dos", "em", "no", "na", "nos", "nas", "e", "ou", "que", "se", "para",
    "por", "com", "ao", "aos", "sem", "sobre", "qual", "quais", "quem",
    "como", "onde", "quando", "e", "sao", "ser", "foi", "segundo", "pela",
    "pelo", "pelas", "pelos", "este", "esta", "isso", "esse", "essa", "seu",
    "sua", "seus", "suas", "mais", "mas", "ja", "tambem", "entre",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _stem(token: str) -> str:
    """Stemming bem simples: remove plural em -s/-es para tokens longos."""
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(_strip_accents(text.lower()))
    return [_stem(t) for t in tokens if t not in _STOPWORDS]


class HashingEmbedder:
    """Embedder lexical (unigramas + bigramas) com hashing assinado."""

    def __init__(self, dim: int = 1024, bigram_weight: float = 0.6):
        self.model_name = "hashing-baseline-offline"
        self._dim = dim
        self._bigram_weight = bigram_weight

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = _tokenize(text)

        counts: dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        for a, b in zip(tokens, tokens[1:]):
            key = f"{a}_{b}"
            counts[key] = counts.get(key, 0.0) + self._bigram_weight

        for feature, count in counts.items():
            h = zlib.crc32(feature.encode("utf-8"))
            idx = h % self._dim
            sign = 1.0 if (h >> 31) & 1 else -1.0  # hashing assinado
            vec[idx] += sign * (1.0 + np.log(count)) if count >= 1 else sign * count

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: list[str], batch_size: int = 32,
                    show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.array([])
        return np.vstack([self._vectorize(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vectorize(query)
