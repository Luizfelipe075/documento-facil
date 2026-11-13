"""
src/embeddings/embedder.py

Geração de embeddings multilíngues para textos em português brasileiro.

Modelo padrão: intfloat/multilingual-e5-large
- 768 dimensões
- Suporte excelente a português
- Modelo local (sem custo de API)
- Protocolo E5: chunks devem ser prefixados com 'passage: '
                queries devem ser prefixadas com 'query: '

Referência: https://huggingface.co/intfloat/multilingual-e5-large
"""

import logging
import os
import time
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")


class Embedder:
    """
    Wrapper para geração de embeddings com sentence-transformers.

    O modelo E5 requer prefixos específicos:
    - Textos de passagem (chunks): 'passage: ' + texto
    - Queries do usuário: 'query: ' + texto

    Isso é parte do protocolo do modelo e não deve ser omitido.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = self._load_model()

    def _load_model(self):
        """Carrega o modelo de embeddings com tratamento de erro."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Carregando modelo de embeddings: %s", self.model_name)
            start = time.monotonic()
            model = SentenceTransformer(self.model_name)
            elapsed = time.monotonic() - start
            logger.info("Modelo carregado em %.1fs", elapsed)
            return model
        except ImportError:
            raise ImportError(
                "sentence-transformers não instalado. "
                "Execute: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error("Erro ao carregar modelo %s: %s", self.model_name, e)
            raise

    @property
    def embedding_dim(self) -> int:
        """Dimensão dos vetores gerados pelo modelo."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Gera embeddings para uma lista de textos (chunks/passagens).
        Aplica o prefixo 'passage: ' exigido pelo protocolo E5.

        Args:
            texts: Lista de textos a embedar.
            batch_size: Tamanho do batch para processamento.
            show_progress: Exibe barra de progresso.

        Returns:
            Array numpy de shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        # Aplica prefixo do protocolo E5
        prefixed = [f"passage: {t}" for t in texts]

        start = time.monotonic()
        logger.info("Gerando embeddings para %d textos...", len(texts))

        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # normaliza para similaridade cosseno
            convert_to_numpy=True,
        )

        elapsed = time.monotonic() - start
        logger.info(
            "Embeddings gerados: %d vetores x %d dims em %.1fs",
            len(embeddings), self.embedding_dim, elapsed,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Gera embedding para uma query do usuário.
        Aplica o prefixo 'query: ' exigido pelo protocolo E5.

        Args:
            query: Pergunta do usuário.

        Returns:
            Array numpy de shape (embedding_dim,).
        """
        prefixed_query = f"query: {query}"
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding
