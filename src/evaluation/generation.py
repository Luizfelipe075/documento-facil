"""
src/evaluation/generation.py

Avaliação da qualidade das respostas geradas pelo LLM.

Critérios de avaliação:
- Correta: A resposta está de acordo com o documento?
- Relevante: A resposta responde à pergunta?
- Fundamentada: A resposta é sustentada pelos trechos recuperados?
- Fonte correta: A fonte indicada corresponde à informação?
- Alucinação: O modelo inventou alguma informação?

A avaliação é registrada manualmente ou por um LLM avaliador.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationRow:
    """Linha de avaliação para uma pergunta."""
    question: str
    answer: str
    sources: list[str]
    correta: Optional[bool] = None        # avaliado manualmente
    relevante: Optional[bool] = None
    fundamentada: Optional[bool] = None
    fonte_correta: Optional[bool] = None
    alucinacao: Optional[bool] = None
    notas: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_evaluation(
    questions_file: Path,
    output_file: Path,
    vectorstore_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Executa o pipeline RAG para todas as perguntas e salva os resultados
    em um arquivo JSON para avaliação manual posterior.

    Args:
        questions_file: Caminho para test_questions.json
        output_file: Onde salvar os resultados (JSON).
        vectorstore_dir: Diretório do ChromaDB.

    Returns:
        Lista de dicionários com perguntas e respostas.
    """
    from src.generation.llm import answer_question

    if vectorstore_dir is None:
        vectorstore_dir = Path("vectorstore")

    output_file = Path(output_file)

    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for i, q in enumerate(questions, start=1):
        question = q["question"]
        logger.info("[%d/%d] Avaliando: %r", i, len(questions), question)

        try:
            result = answer_question(question, vectorstore_dir=vectorstore_dir)
            source_urls = [s["source_url"] for s in result["sources"]]
            row = EvaluationRow(
                question=question,
                answer=result["answer"],
                sources=source_urls,
            ).to_dict()
        except Exception as e:
            logger.error("Erro ao processar pergunta %r: %s", question, e)
            row = EvaluationRow(
                question=question,
                answer=f"ERRO: {e}",
                sources=[],
                notas="Erro durante execução",
            ).to_dict()

        rows.append(row)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    logger.info("Avaliação salva em: %s", output_file)
    return rows


def print_evaluation_table(rows: list[dict]) -> None:
    """Imprime uma tabela resumida de avaliação no terminal."""
    header = f"{'Pergunta':<50} {'Correta':>8} {'Relevante':>10} {'Fundamentada':>13} {'Fonte OK':>9} {'Alucinação':>11}"
    print(header)
    print("-" * len(header))
    for row in rows:
        q = row["question"][:48] + ".." if len(row["question"]) > 50 else row["question"]
        correta = "✓" if row.get("correta") else ("✗" if row.get("correta") is False else "?")
        relevante = "✓" if row.get("relevante") else ("✗" if row.get("relevante") is False else "?")
        fundamentada = "✓" if row.get("fundamentada") else ("✗" if row.get("fundamentada") is False else "?")
        fonte = "✓" if row.get("fonte_correta") else ("✗" if row.get("fonte_correta") is False else "?")
        alucina = "Não" if row.get("alucinacao") is False else ("Sim" if row.get("alucinacao") else "?")
        print(f"{q:<50} {correta:>8} {relevante:>10} {fundamentada:>13} {fonte:>9} {alucina:>11}")
