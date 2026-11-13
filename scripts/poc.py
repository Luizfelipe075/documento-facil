"""
scripts/poc.py

Proof of Concept (Fase 1) - Documento Facil.
Demonstra o pipeline completo com um unico documento:

    PDF/HTML -> extracao -> limpeza -> chunking -> embedding -> indice -> resposta

Uso:
    python scripts/poc.py --demo
    python scripts/poc.py --file data/raw/html/meu.html --question "Qual e o tema?"
    python scripts/poc.py --file data/raw/pdf/lei.pdf  --question "O que esta lei estabelece?"
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = ROOT / "vectorstore"


def separator(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def run_poc(file_path: Path, question: str, document_id: str = "poc_doc") -> None:
    """
    Pipeline completo para um unico documento.
    """
    from dotenv import load_dotenv
    load_dotenv()

    separator("DOCUMENTO FACIL - Proof of Concept")
    print(f"  Arquivo : {file_path}")
    print(f"  Pergunta: {question}")
    separator()

    # -----------------------------------------------------------
    # 1. EXTRACAO
    # -----------------------------------------------------------
    print("\n[1/7] Extraindo texto...")
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from src.extraction.pdf import extract_pdf
        result = extract_pdf(file_path, document_id)
        if result is None:
            print("ERRO: nao foi possivel extrair o PDF.")
            return
        if not result.has_text:
            print("AVISO: PDF possivelmente digitalizado. Considere OCR.")
        raw_text = result.full_text
        metadata = {
            "document_id": document_id,
            "title": file_path.stem,
            "source_url": file_path.as_uri(),
            "organization": "Documento local",
            "publication_date": None,
        }

    elif suffix in (".html", ".htm"):
        from src.extraction.html import extract_html
        result = extract_html(file_path, document_id, file_path.as_uri())
        if result is None:
            print("ERRO: nao foi possivel extrair o HTML.")
            return
        raw_text = result.body
        metadata = {
            "document_id": document_id,
            "title": result.title or file_path.stem,
            "source_url": file_path.as_uri(),
            "organization": result.organization or "Documento local",
            "publication_date": result.date,
        }

    else:
        print(f"ERRO: formato nao suportado: {suffix}. Use .pdf ou .html")
        return

    print(f"   Extraidos: {len(raw_text):,} caracteres")

    # -----------------------------------------------------------
    # 2. LIMPEZA
    # -----------------------------------------------------------
    print("\n[2/7] Limpando e normalizando texto...")
    from src.processing.cleaner import clean_text
    clean = clean_text(raw_text, document_id)
    print(f"   Apos limpeza: {len(clean):,} caracteres")

    # -----------------------------------------------------------
    # 3. CHUNKING
    # -----------------------------------------------------------
    print("\n[3/7] Dividindo em chunks (500 tokens, overlap 100)...")
    from src.processing.chunker import chunk_document
    chunks = chunk_document(clean, document_id, metadata, chunk_size=500, overlap=100)
    print(f"   Chunks gerados: {len(chunks)}")

    # -----------------------------------------------------------
    # 4. EMBEDDINGS
    # -----------------------------------------------------------
    print("\n[4/7] Carregando modelo de embeddings...")
    from src.embeddings.embedder import Embedder
    embedder = Embedder()
    print(f"   Modelo: {embedder.model_name}")
    print(f"   Dimensoes: {embedder.embedding_dim}")

    # -----------------------------------------------------------
    # 5. INDEXACAO
    # -----------------------------------------------------------
    print("\n[5/7] Indexando no ChromaDB...")
    from src.retrieval.index import VectorIndex
    index = VectorIndex(persist_dir=VECTORSTORE_DIR, embedder=embedder)
    index.build(chunks, embedder=embedder)
    print(f"   Chunks no indice: {index.count}")

    # -----------------------------------------------------------
    # 6. BUSCA SEMANTICA
    # -----------------------------------------------------------
    print("\n[6/7] Buscando trechos relevantes...")
    query_vec = embedder.embed_query(question)
    results = index.search(query_vec, k=5)
    print(f"   Encontrados {len(results)} trechos:")
    for i, r in enumerate(results, 1):
        score = r["score"]
        idx = r["chunk_index"]
        preview = r["text"][:80].replace("\n", " ")
        print(f"   [{i}] Score={score:.3f} Chunk#{idx}: {preview}...")

    # -----------------------------------------------------------
    # 7. GERACAO DE RESPOSTA
    # -----------------------------------------------------------
    print("\n[7/7] Gerando resposta com LLM...")
    from src.retrieval.search import reset_singletons
    from src.generation.llm import answer_question
    reset_singletons()

    output = answer_question(question, k=5, vectorstore_dir=VECTORSTORE_DIR)

    separator("RESPOSTA")
    print(output["answer"])

    if output["sources"]:
        print("\nFONTES:")
        for src in output["sources"]:
            title = src.get("title", "Sem titulo")
            url = src.get("source_url", "")
            print(f"  - {title}")
            if url:
                print(f"    {url}")

    separator()
    print("Pipeline POC concluido com sucesso.")


def demo_mode() -> None:
    """
    Modo demonstracao: baixa a LDB (Lei de Diretrizes e Bases) do Planalto
    e faz uma pergunta sobre ela.
    """
    import requests

    demo_url = "https://www.planalto.gov.br/ccivil_03/leis/l9394.htm"
    demo_file = ROOT / "data" / "raw" / "html" / "poc_ldb_demo.html"

    print(f"Baixando documento de demonstracao...")
    print(f"URL: {demo_url}")

    headers = {"User-Agent": "DocumentoFacil/1.0 (projeto academico)"}
    try:
        response = requests.get(demo_url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"ERRO ao baixar documento: {e}")
        print("Verifique sua conexao com a internet e tente novamente.")
        sys.exit(1)

    demo_file.parent.mkdir(parents=True, exist_ok=True)
    with open(demo_file, "w", encoding="utf-8", errors="replace") as f:
        f.write(response.text)
    print(f"Documento salvo em: {demo_file}")

    run_poc(
        file_path=demo_file,
        question="Quais sao os principios e fins da educacao?",
        document_id="poc_ldb_l9394",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Documento Facil - Proof of Concept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Modo demo (baixa a LDB do Planalto):
  python scripts/poc.py --demo

  # Com arquivo local:
  python scripts/poc.py --file data/raw/html/lei.html --question "Qual e o objeto da lei?"
  python scripts/poc.py --file data/raw/pdf/decreto.pdf --question "O que este decreto regulamenta?"
        """,
    )
    parser.add_argument("--file", type=Path, help="Caminho para arquivo PDF ou HTML")
    parser.add_argument("--question", type=str, help="Pergunta a responder")
    parser.add_argument("--demo", action="store_true", help="Modo demo (baixa LDB do Planalto)")
    parser.add_argument("--doc-id", type=str, default="poc_doc", help="ID do documento")

    args = parser.parse_args()

    if args.demo:
        demo_mode()
    elif args.file and args.question:
        if not args.file.exists():
            print(f"ERRO: arquivo nao encontrado: {args.file}")
            sys.exit(1)
        run_poc(args.file, args.question, args.doc_id)
    else:
        parser.print_help()
        sys.exit(1)
