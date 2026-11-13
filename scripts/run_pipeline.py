"""
scripts/run_pipeline.py

Pipeline completo do Documento Facil (Fases 2 e 3).
Executa: coleta -> extracao -> limpeza -> chunking -> embeddings -> indice.

Uso:
    python scripts/run_pipeline.py                    # coleta do Planalto (limite 50)
    python scripts/run_pipeline.py --limit 10         # coleta 10 documentos
    python scripts/run_pipeline.py --skip-collect     # so processa docs ja baixados
    python scripts/run_pipeline.py --reset-index      # reinicia o indice vetorial
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
VECTORSTORE_DIR = ROOT / "vectorstore"
DOCUMENTS_FILE = DATA_DIR / "processed" / "documents.jsonl"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"


def separator(msg: str = "") -> None:
    line = "-" * 60
    if msg:
        logger.info(line)
        logger.info("  %s", msg)
        logger.info(line)
    else:
        logger.info(line)


def step_collect(limit: int) -> list[dict]:
    """
    Fase 2: Coleta de documentos do Portal do Planalto.
    """
    separator("ETAPA 1/5: COLETA")
    from src.collectors.planalto import PlanaltoCollector

    collector = PlanaltoCollector(data_dir=DATA_DIR, rate_limit_seconds=1.5)
    docs = collector.collect(limit=limit)
    logger.info("Coleta concluida: %d novos documentos.", len(docs))
    return docs


def step_extract(documents_file: Path) -> list[dict]:
    """
    Fase 3a: Extracao de texto de todos os documentos coletados.
    Processa PDFs e HTMLs armazenados em data/raw/.
    """
    separator("ETAPA 2/5: EXTRACAO")
    from src.extraction.pdf import extract_pdf
    from src.extraction.html import extract_html

    if not documents_file.exists():
        logger.error("Arquivo de metadados nao encontrado: %s", documents_file)
        logger.error("Execute a coleta antes da extracao.")
        return []

    extracted = []
    with open(documents_file, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    logger.info("Processando %d documentos para extracao...", len(docs))

    for doc in docs:
        doc_id = doc["document_id"]
        content_type = doc.get("content_type", "html")
        file_path_str = doc.get("file_path", "")

        if not file_path_str:
            logger.warning("Sem file_path para %s, pulando.", doc_id)
            continue

        file_path = ROOT / file_path_str

        if not file_path.exists():
            logger.warning("Arquivo nao encontrado: %s", file_path)
            continue

        try:
            if content_type == "pdf":
                result = extract_pdf(file_path, doc_id)
                if result is None or not result.full_text.strip():
                    logger.warning("PDF sem texto extraivel: %s", doc_id)
                    continue
                text = result.full_text
            else:
                source_url = doc.get("source_url", "")
                result = extract_html(file_path, doc_id, source_url)
                if result is None or not result.body.strip():
                    logger.warning("HTML sem conteudo: %s", doc_id)
                    continue
                # Atualiza metadados com informacoes extraidas do HTML
                if result.title and not doc.get("title"):
                    doc["title"] = result.title
                if result.date and not doc.get("publication_date"):
                    doc["publication_date"] = result.date
                text = result.body

            extracted.append({"doc": doc, "text": text})

        except Exception as e:
            logger.error("Erro ao extrair %s: %s", doc_id, e)

    logger.info("Extracao concluida: %d documentos com texto.", len(extracted))
    return extracted


def step_process(extracted: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    Fase 3b: Limpeza, normalizacao e chunking.
    """
    separator("ETAPA 3/5: LIMPEZA E CHUNKING")
    from src.processing.cleaner import clean_text
    from src.processing.chunker import chunk_document

    all_chunks = []

    for item in extracted:
        doc = item["doc"]
        raw_text = item["text"]
        doc_id = doc["document_id"]

        try:
            clean = clean_text(raw_text, doc_id)
            if not clean.strip():
                logger.warning("Texto vazio apos limpeza: %s", doc_id)
                continue

            chunks = chunk_document(
                text=clean,
                document_id=doc_id,
                metadata=doc,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error("Erro ao processar %s: %s", doc_id, e)

    logger.info("Chunking concluido: %d chunks totais.", len(all_chunks))

    # Persiste chunks em JSONL
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False, default=str) + "\n")
    logger.info("Chunks salvos em: %s", CHUNKS_FILE)

    return all_chunks


def step_index(chunks: list[dict], reset: bool = False) -> None:
    """
    Fases 4 e 5: Geracao de embeddings e indexacao no ChromaDB.
    """
    separator("ETAPA 4/5: EMBEDDINGS")
    from src.embeddings.embedder import Embedder

    embedder = Embedder()
    logger.info("Modelo: %s (%d dims)", embedder.model_name, embedder.embedding_dim)

    separator("ETAPA 5/5: INDEXACAO NO CHROMADB")
    from src.retrieval.index import VectorIndex

    index = VectorIndex(persist_dir=VECTORSTORE_DIR, embedder=embedder)

    if reset:
        logger.warning("Resetando indice vetorial...")
        index.reset()

    index.build(chunks, embedder=embedder)
    logger.info("Indexacao concluida. Total no indice: %d chunks.", index.count)


def load_existing_chunks() -> list[dict]:
    """Carrega chunks ja processados do arquivo JSONL."""
    if not CHUNKS_FILE.exists():
        return []
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main(limit: int, skip_collect: bool, skip_extract: bool, reset_index: bool) -> None:
    separator("DOCUMENTO FACIL - Pipeline Completo")
    logger.info("Limite de coleta: %d documentos", limit)
    logger.info("Diretorio de dados: %s", DATA_DIR)
    logger.info("Diretorio do indice: %s", VECTORSTORE_DIR)
    separator()

    all_chunks = []

    if skip_extract and CHUNKS_FILE.exists():
        # Pula coleta e extracao, usa chunks ja existentes
        logger.info("--skip-extract: carregando chunks existentes de %s", CHUNKS_FILE)
        all_chunks = load_existing_chunks()
        logger.info("Chunks carregados: %d", len(all_chunks))
    else:
        # Etapa 1: Coleta
        if not skip_collect:
            step_collect(limit=limit)
        else:
            logger.info("--skip-collect: pulando coleta.")

        # Etapa 2: Extracao
        extracted = step_extract(DOCUMENTS_FILE)
        if not extracted:
            logger.error("Nenhum documento extraido. Verifique os logs.")
            sys.exit(1)

        # Etapa 3: Limpeza + Chunking
        all_chunks = step_process(extracted, chunk_size=500, overlap=100)

    if not all_chunks:
        logger.error("Nenhum chunk gerado. Pipeline encerrado.")
        sys.exit(1)

    # Etapas 4+5: Embeddings + Indice
    step_index(all_chunks, reset=reset_index)

    separator("PIPELINE CONCLUIDO")
    logger.info("Execute a interface com: streamlit run app/streamlit_app.py")
    separator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Documento Facil - Pipeline completo de coleta e indexacao",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Pipeline completo (coleta + processa + indexa):
  python scripts/run_pipeline.py

  # Coletar apenas 10 documentos:
  python scripts/run_pipeline.py --limit 10

  # Pular coleta (reprocessar docs ja baixados):
  python scripts/run_pipeline.py --skip-collect

  # Pular tudo e so reindexar (usando chunks.jsonl existente):
  python scripts/run_pipeline.py --skip-extract --reset-index
        """,
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximo de documentos a coletar (padrao: 50)")
    parser.add_argument("--skip-collect", action="store_true", help="Pula a etapa de coleta")
    parser.add_argument("--skip-extract", action="store_true", help="Usa chunks.jsonl existente, pula coleta e extracao")
    parser.add_argument("--reset-index", action="store_true", help="Reinicia o indice vetorial antes de indexar")

    args = parser.parse_args()
    main(
        limit=args.limit,
        skip_collect=args.skip_collect,
        skip_extract=args.skip_extract,
        reset_index=args.reset_index,
    )
