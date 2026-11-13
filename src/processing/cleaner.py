"""
src/processing/cleaner.py

Pipeline de limpeza e normalização de texto para documentos públicos brasileiros.

CRITÉRIO PARA REMOÇÃO DE ELEMENTOS REPETITIVOS:
Um elemento é considerado ruído repetitivo se aparecer mais de 3 vezes
no texto com o mesmo conteúdo E tiver menos de 100 caracteres.
Isso captura cabeçalhos e rodapés de página sem remover conteúdo jurídico
que possa repetir termos (como 'Art. 1º' em leis diferentes).

PRESERVAÇÃO OBRIGATÓRIA:
- Artigos (Art. 1º, Art. 2º, ...)
- Parágrafos (§ 1º, § 2º, ...)
- Incisos (I -, II -, III -, ...)
- Alíneas (a), b), c), ...)
- Datas e números
- Nomes próprios
- Referências legais (Lei nº X, Decreto nº Y, ...)
"""

import logging
import re
import unicodedata
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Comprimento máximo de uma linha para ser considerada ruído repetitivo.
# Cabeçalhos e rodapés de página são geralmente curtos.
MAX_NOISE_LINE_LENGTH = 120

# Número mínimo de repetições para uma linha ser considerada ruído.
MIN_NOISE_REPETITIONS = 3


def _normalize_unicode(text: str) -> str:
    """
    Normaliza caracteres Unicode para NFC.
    Corrige problemas comuns de codificação em PDFs brasileiros
    (caracteres acentuados mal representados).
    """
    return unicodedata.normalize("NFC", text)


def _remove_separator_lines(text: str) -> str:
    """
    Remove linhas que são apenas separadores gráficos.
    Critério: linha com 3+ caracteres iguais (-, _, *, =, ~).
    Esses separadores são comuns em cabeçalhos de PDFs e não carregam
    significado jurídico.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Linha de separador: apenas um tipo de caractere repetido
        if stripped and re.fullmatch(r"[-_*=~]{3,}", stripped):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_repeated_headers_footers(text: str) -> str:
    """
    Remove cabeçalhos e rodapés repetidos detectados por frequência.

    Critério de remoção:
    - Linha aparece MIN_NOISE_REPETITIONS ou mais vezes no texto.
    - Linha tem menos de MAX_NOISE_LINE_LENGTH caracteres.
    - Linha não começa com indicador jurídico (Art., §, I -, etc.).

    Preservação:
    - Linhas iniciando com marcadores jurídicos NUNCA são removidas,
      mesmo que se repitam, porque repetição é estrutural em leis.
    """
    LEGAL_MARKERS = re.compile(
        r"^(Art\.|§|I{1,3}V?\s[-–]|V?I{0,3}\s[-–]|[a-z]\)|\d+[.º°])",
        re.IGNORECASE,
    )

    lines = text.split("\n")
    line_counts = Counter(line.strip() for line in lines if line.strip())

    noise_lines = set()
    for line, count in line_counts.items():
        if (
            count >= MIN_NOISE_REPETITIONS
            and len(line) <= MAX_NOISE_LINE_LENGTH
            and not LEGAL_MARKERS.match(line)
        ):
            noise_lines.add(line)
            logger.debug("Ruído identificado (%dx): %r", count, line[:60])

    if noise_lines:
        logger.info("%d padrões de ruído detectados e removidos.", len(noise_lines))

    cleaned = []
    for line in lines:
        if line.strip() in noise_lines:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _normalize_whitespace(text: str) -> str:
    """
    Normaliza espaços:
    - Colapsa múltiplos espaços em um único.
    - Remove espaços antes de pontuação.
    - Preserva indentação mínima de alíneas e incisos.
    """
    # Colapsa espaços múltiplos (preserva quebras de linha)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove espaço antes de ponto final, vírgula, ponto e vírgula
    text = re.sub(r" ([.,;:])", r"\1", text)
    return text


def _normalize_line_breaks(text: str) -> str:
    """
    Normaliza quebras de linha:
    - Máximo de 2 linhas em branco consecutivas.
    - Remove linhas que são apenas espaço.
    """
    # Remove espaços em linhas vazias
    text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
    # Colapsa 3+ quebras de linha em 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str, document_id: Optional[str] = None) -> str:
    """
    Pipeline completo de limpeza para textos de documentos públicos brasileiros.

    Etapas:
    1. Normalização Unicode (NFC)
    2. Remoção de linhas separadoras
    3. Remoção de cabeçalhos/rodapés repetidos
    4. Normalização de espaços
    5. Normalização de quebras de linha

    Args:
        raw_text: Texto bruto extraído de PDF ou HTML.
        document_id: Identificador do documento (para logging).

    Returns:
        Texto normalizado preservando a estrutura jurídica.
    """
    doc_label = document_id or "documento"
    original_len = len(raw_text)

    text = _normalize_unicode(raw_text)
    text = _remove_separator_lines(text)
    text = _remove_repeated_headers_footers(text)
    text = _normalize_whitespace(text)
    text = _normalize_line_breaks(text)

    reduction = original_len - len(text)
    logger.info(
        "Limpeza concluída [%s]: %d → %d chars (-%d)",
        doc_label, original_len, len(text), reduction,
    )
    return text
