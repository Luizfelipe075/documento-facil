"""
app/streamlit_app.py

Interface Streamlit do Documento Facil.
Execucao:
    streamlit run app/streamlit_app.py
"""

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = ROOT / "vectorstore"

st.set_page_config(
    page_title="Documento Facil",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .ai-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 16px;
        color: #664d03;
    }
    .source-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Configuracoes")
    api_key = st.text_input(
        "Chave da API Gemini (opcional)",
        type="password",
        help="A chave digitada aqui vale apenas nesta sessão do navegador. Para salvar, configure GOOGLE_API_KEY no arquivo .env.",
        key="gemini_api_key",
    )
    st.caption("A chave não é exibida nem registrada nos logs.")
    st.markdown("---")
    retrieval_k = st.slider("Trechos recuperados (K)", 1, 10, 5)
    show_chunks = st.checkbox("Mostrar trechos recuperados", value=False)
    st.markdown("---")
    st.markdown("### Status do indice")
    if st.button("Verificar indice"):
        try:
            from src.retrieval.index import VectorIndex
            idx = VectorIndex(persist_dir=VECTORSTORE_DIR)
            count = idx.count
            if count > 0:
                st.success(f"Indice ativo: {count} chunks indexados")
            else:
                st.warning("Indice vazio. Execute run_pipeline.py primeiro.")
        except Exception as e:
            st.error(f"Erro: {e}")
    st.markdown("---")
    st.markdown("**Documento Facil** v1.0  \nSistema RAG para documentos publicos brasileiros.")

st.title("Documento Facil")
st.markdown("**IA para organizacao inteligente de documentos de interesse publico**")
st.markdown("---")

question = st.text_area(
    "Pergunte sobre documentos publicos:",
    placeholder=(
        "Ex: O que estabelece a Lei no 14.133 sobre licitacoes?\n"
        "Ex: Quais sao os principios da administracao publica?\n"
        "Ex: Qual e a data de publicacao da Lei de Responsabilidade Fiscal?"
    ),
    height=100,
    key="question_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    search_btn = st.button("Pesquisar", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Limpar", use_container_width=True)

if clear_btn:
    st.rerun()

if search_btn and question.strip():
    with st.status("Iniciando pesquisa…", expanded=True) as progress:
        try:
            from src.generation.llm import answer_question
            result = answer_question(
                question=question.strip(),
                k=retrieval_k,
                vectorstore_dir=VECTORSTORE_DIR,
                api_key=api_key.strip() or None,
                progress_callback=lambda message: progress.update(label=message),
                provider="gemini",
            )
            st.session_state["result"] = result
            progress.update(label="Resposta pronta", state="complete", expanded=False)
        except Exception as e:
            logger.exception("Erro ao processar pergunta")
            progress.update(label="A pesquisa falhou", state="error", expanded=True)
            st.error(f"Erro: {e}")
            st.info(
                "Verifique se:\n"
                "1. O arquivo .env esta configurado com a chave de API.\n"
                "2. O indice foi construido (execute scripts/run_pipeline.py).\n"
                "3. As dependencias estao instaladas."
            )
elif search_btn and not question.strip():
    st.warning("Por favor, digite uma pergunta antes de pesquisar.")

if "result" in st.session_state:
    result = st.session_state["result"]
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    raw_chunks = result.get("raw_chunks", [])

    st.markdown("## Resposta")
    st.markdown(answer)

    if sources:
        st.markdown("---")
        st.markdown("## Fontes utilizadas")
        for source in sources:
            title = source.get("title") or "Documento sem titulo"
            url = source.get("source_url", "")
            org = source.get("organization", "")
            date = source.get("publication_date", "")
            with st.container():
                st.markdown(f"**{title}**")
                if org:
                    st.caption(f"Órgão: {org}")
                if date:
                    st.caption(f"Data: {date}")
                if url:
                    st.markdown(f"[Abrir documento oficial]({url})")
                else:
                    st.caption("URL não disponível")
    else:
        st.info("Nenhuma fonte identificada para esta resposta.")

    if show_chunks and raw_chunks:
        st.markdown("---")
        st.markdown("## Trechos recuperados")
        st.caption(f"{len(raw_chunks)} trecho(s) utilizados.")
        for i, chunk in enumerate(raw_chunks, start=1):
            score = chunk.get("score", 0)
            chunk_title = chunk.get("title") or "Sem titulo"
            chunk_url = chunk.get("source_url", "")
            chunk_text = chunk.get("text", "")
            with st.expander(f"Trecho {i} | {chunk_title[:60]} | Relevancia: {score:.2f}", expanded=False):
                if chunk_url:
                    st.markdown(f"[Fonte original]({chunk_url})")
                st.text(chunk_text)

    st.markdown("---")
    st.markdown(
        '<div class="ai-warning">'
        "<strong>Aviso:</strong> Esta resposta foi produzida por <strong>Inteligencia Artificial</strong> "
        "com base nos documentos publicos recuperados. Nao constitui orientacao juridica. "
        "Consulte sempre o documento oficial para verificar o conteudo integral."
        "</div>",
        unsafe_allow_html=True,
    )
