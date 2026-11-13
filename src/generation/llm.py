"""
src/generation/llm.py

Geração de respostas usando LLM (Gemini ou OpenAI).
Implementa o pipeline RAG completo: search → context → LLM → resposta + fontes.

O LLM nunca recebe o documento inteiro:
apenas os trechos recuperados pela busca semântica.

O sistema é configurado para responder exclusivamente com base
nos documentos recuperados, sem inventar informações.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# Prompt do sistema: define o comportamento do assistente
SYSTEM_PROMPT = """Você é um assistente de pesquisa de documentos públicos brasileiros.

Sua função é responder perguntas utilizando EXCLUSIVAMENTE os trechos fornecidos no contexto.

REGRAS:
1. Não invente informações.
2. Não utilize conhecimento externo ao contexto.
3. Se o contexto não for suficiente, informe que não foi possível encontrar informação suficiente.
4. Responda em português brasileiro.
5. Seja objetivo.
6. Explique conceitos jurídicos ou administrativos em linguagem simples quando possível, sem alterar seu significado.
7. Diferencie claramente informações presentes em documentos diferentes.
8. Sempre indique a fonte utilizada.
9. Sempre forneça o título do documento.
10. Sempre forneça a URL do documento original.
11. Quando disponível, informe a página ou trecho de origem.
12. Não apresente a resposta como orientação jurídica.
13. Informe que a resposta foi produzida por IA.

FORMATO DA RESPOSTA:

Resposta:
[resposta resumida baseada exclusivamente nos documentos]

Fontes:
- [título do documento] — trecho [N]
  [URL do documento]

Aviso:
Esta resposta foi produzida por Inteligência Artificial com base nos documentos públicos recuperados pelo sistema. Consulte o documento oficial para verificar o conteúdo integral."""


def _build_context(chunks: list[dict]) -> str:
    """
    Formata os chunks recuperados em um contexto estruturado para o LLM.
    Cada trecho é numerado e inclui referência ao documento original.
    """
    if not chunks:
        return "Nenhum trecho relevante encontrado nos documentos."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.get("title") or "Documento sem título"
        url = chunk.get("source_url") or "URL não disponível"
        org = chunk.get("organization") or ""
        date = chunk.get("publication_date") or ""
        text = chunk.get("text", "")
        score = chunk.get("score", 0)

        header = f"[Trecho {i}] {title}"
        if org:
            header += f" | {org}"
        if date:
            header += f" | {date}"
        header += f" | Relevância: {score:.2f}"
        header += f"\nURL: {url}"

        parts.append(f"{header}\n\n{text}")

    separator = "\n\n" + "=" * 60 + "\n"
    return separator + separator.join(parts)


def _build_user_message(question: str, context: str) -> str:
    """Monta a mensagem do usuário com a pergunta e o contexto."""
    return f"""PERGUNTA DO USUÁRIO:
{question}

CONTEXTO — DOCUMENTOS RECUPERADOS:
{context}

Responda à pergunta baseando-se exclusivamente nos documentos acima."""


class GeminiClient:
    """Cliente para a API do Google Gemini."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.api_key = api_key
        self._client = None
        self._init()

    def _init(self):
        try:
            from google import genai
            from google.genai import types
            api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Chave Gemini ausente. Cole a chave na barra lateral do site "
                    "ou configure GOOGLE_API_KEY no arquivo .env."
                )
            self._client = genai.Client(api_key=api_key)
            self._types = types
            logger.info("Gemini inicializado: %s", self.model_name)
        except ImportError:
            raise ImportError(
                "google-genai não instalado. Execute: pip install google-genai"
            )

    def generate(self, user_message: str) -> str:
        """Envia mensagem e retorna resposta textual."""
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            if not response.text:
                raise RuntimeError("O Gemini retornou uma resposta vazia.")
            return response.text
        except Exception as e:
            logger.error("Erro na geração Gemini: %s", e)
            raise


class OpenAIClient:
    """Cliente para a API da OpenAI."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key
        self._init()

    def _init(self):
        try:
            import openai
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY não encontrada no .env")
            self._client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI inicializado: %s", self.model_name)
        except ImportError:
            raise ImportError(
                "openai não instalado. Execute: pip install openai"
            )

    def generate(self, user_message: str) -> str:
        """Envia mensagem e retorna resposta textual."""
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,  # baixa criatividade para mais fidelidade
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("Erro na geração OpenAI: %s", e)
            raise


def _get_llm_client(api_key: Optional[str] = None, provider: Optional[str] = None):
    """Instancia o cliente LLM conforme configuração no .env."""
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider not in {"gemini", "openai"}:
        raise ValueError("LLM_PROVIDER deve ser 'gemini' ou 'openai'.")
    if provider == "openai":
        return OpenAIClient(api_key=api_key)
    return GeminiClient(api_key=api_key)


def answer_question(
    question: str,
    k: int = 5,
    vectorstore_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    progress_callback=None,
    provider: Optional[str] = None,
) -> dict:
    """
    Pipeline RAG completo: pergunta → busca → contexto → LLM → resposta.

    Args:
        question: Pergunta do usuário em linguagem natural.
        k: Número de chunks a recuperar.
        vectorstore_dir: Diretório do ChromaDB.

    Returns:
        {
            'question': str,
            'answer': str,
            'sources': list[dict],  # chunks utilizados
            'raw_chunks': list[dict],
        }
    """
    from src.retrieval.search import search

    if vectorstore_dir is None:
        vectorstore_dir = Path("vectorstore")

    logger.info("Respondendo pergunta: %r", question)

    # 1. Recuperar chunks relevantes
    logger.info("ETAPA 1/3: Gerando embedding da pergunta e buscando no índice")
    if progress_callback:
        progress_callback("Buscando trechos relevantes no índice local…")
    chunks = search(question, k=k, vectorstore_dir=vectorstore_dir)

    if not chunks:
        no_info_msg = (
            "Não encontrei informações suficientes nos documentos consultados "
            "para responder com segurança.\n\n"
            "Aviso: Esta resposta foi produzida por Inteligência Artificial. "
            "O índice pode estar vazio ou não conter documentos sobre este tema."
        )
        return {
            "question": question,
            "answer": no_info_msg,
            "sources": [],
            "raw_chunks": [],
        }

    # 2. Construir contexto
    logger.info("ETAPA 2/3: Preparando contexto com %d trechos", len(chunks))
    context = _build_context(chunks)

    # 3. Montar mensagem
    user_message = _build_user_message(question, context)

    # 4. Gerar resposta
    logger.info("ETAPA 3/3: Enviando contexto ao provedor LLM configurado")
    if progress_callback:
        progress_callback("Trechos encontrados. Gerando resposta com Gemini…")
    llm = _get_llm_client(api_key=api_key, provider=provider)
    answer = llm.generate(user_message)

    # 5. Preparar fontes únicas para exibição
    seen_docs = set()
    sources = []
    for chunk in chunks:
        doc_id = chunk.get("document_id", "")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "title": chunk.get("title") or "Sem título",
                "source_url": chunk.get("source_url", ""),
                "organization": chunk.get("organization", ""),
                "publication_date": chunk.get("publication_date", ""),
            })

    logger.info("Resposta gerada. Fontes: %d documentos únicos.", len(sources))

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "raw_chunks": chunks,
    }
