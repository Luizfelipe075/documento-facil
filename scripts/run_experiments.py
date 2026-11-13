"""
scripts/run_experiments.py

TASK 17 — Executa os experimentos de avaliação do Documento Fácil.

Compara configurações de chunking (tamanho x sobreposição) medindo Recall@K e
MRR da recuperação, e, opcionalmente, avalia a qualidade das respostas geradas.

Pré-requisito: ter documentos coletados. Qualquer um destes serve:
    python scripts/poc.py --demo           # baixa a LDB (1 documento)
    python scripts/run_pipeline.py         # coleta vários documentos

Uso:
    python scripts/run_experiments.py                     # grade padrão, embedder e5 real
    python scripts/run_experiments.py --quick             # 2 configurações (rápido)
    python scripts/run_experiments.py --check-evidence    # valida as evidências no corpus
    python scripts/run_experiments.py --generation        # + respostas do LLM (precisa de chave)
    python scripts/run_experiments.py --embedder hashing  # offline, sem baixar o modelo
    python scripts/run_experiments.py --demo-fixture      # teste de fumaça 100% offline

Saídas (em data/experiments/<data_hora>/):
    report.md, results.json, summary.csv
    generation_results.json, generation_manual_eval.csv   (com --generation)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("run_experiments")

FIXTURE_DIR = ROOT / "tests" / "fixtures"


def build_embedder(kind: str):
    if kind == "hashing":
        from src.embeddings.offline import HashingEmbedder
        return HashingEmbedder()
    try:
        from src.embeddings.embedder import Embedder
        return Embedder()
    except Exception as e:  # noqa: BLE001
        logger.error("Não foi possível carregar o modelo de embeddings: %s", e)
        logger.error("Sem internet/instalação? Use --embedder hashing para rodar offline "
                     "(linha de base lexical, não representa o e5).")
        sys.exit(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Documento Fácil — experimentos de avaliação (TASK 17)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--questions", type=Path, default=ROOT / "data" / "questions" / "test_questions.json")
    p.add_argument("--evidence", type=Path, default=ROOT / "data" / "questions" / "evidence_ldb.json",
                   help="Evidências por pergunta (métricas no nível do trecho)")
    p.add_argument("--aliases", type=Path, default=ROOT / "data" / "questions" / "document_aliases.json")
    p.add_argument("--corpus-dir", type=Path, help="Usa SOMENTE os .html/.pdf desta pasta como corpus")
    p.add_argument("--out-dir", type=Path, help="Pasta de saída (padrão: data/experiments/<data_hora>)")
    p.add_argument("--embedder", choices=["e5", "hashing"], default="e5",
                   help="e5 = multilingual-e5-large (real); hashing = baseline offline")
    p.add_argument("--chunk-sizes", type=int, nargs="+", help="Tamanhos de chunk em tokens (padrão: 250 500 800)")
    p.add_argument("--overlaps", type=int, nargs="+", help="Sobreposições em tokens (padrão: 0 100)")
    p.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5], help="Valores de K (padrão: 1 3 5)")
    p.add_argument("--quick", action="store_true", help="Grade reduzida: tamanhos 300 e 500, sobreposição 100")
    p.add_argument("--generation", action="store_true", help="Avalia também a geração de respostas (usa o LLM)")
    p.add_argument("--generation-config", help="Config para a geração (padrão: a melhor). Ex.: cs500_ov100")
    p.add_argument("--k-generation", type=int, default=5, help="Trechos enviados ao LLM (padrão: 5)")
    p.add_argument("--sleep", type=float, default=4.0, help="Pausa entre chamadas ao LLM, em segundos (padrão: 4)")
    p.add_argument("--check-evidence", action="store_true",
                   help="Só valida se as evidências existem no corpus e sai")
    p.add_argument("--demo-fixture", action="store_true",
                   help="Usa o documento sintético de tests/fixtures (teste de fumaça; implica --embedder hashing)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Bibliotecas ruidosas
    for noisy in ("chromadb", "httpx", "urllib3", "sentence_transformers", "src.retrieval.index",
                  "src.embeddings.embedder", "src.processing.chunker", "src.processing.cleaner",
                  "src.extraction.html", "src.extraction.pdf"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from src.evaluation import experiments as X

    corpus_dir = args.corpus_dir
    embedder_kind = args.embedder
    if args.demo_fixture:
        corpus_dir = FIXTURE_DIR
        embedder_kind = "hashing"
        logger.warning("MODO DEMO: corpus sintético + embedder offline. Resultados só validam o pipeline.")

    # ---- Modo: validar evidências (não precisa de embedder) ----
    if args.check_evidence:
        questions = X.load_questions(args.questions)
        evidence = X.load_evidence(args.evidence, questions)
        corpus = X.load_corpus(ROOT, corpus_dir=corpus_dir)
        if not corpus:
            logger.error("Corpus vazio. Rode `python scripts/poc.py --demo` primeiro.")
            return 1
        report = X.check_evidence(corpus, questions, evidence)
        bad = [r for r in report if not r["ok"]]
        for r in report:
            print(f"[{'OK ' if r['ok'] else 'FALTA'}] {r['id']}: {r['question']}")
            if not r["ok"]:
                print(f"        nenhuma destas alternativas existe no corpus: {r['alternatives']}")
        print(f"\n{len(report) - len(bad)}/{len(report)} evidências encontradas no corpus.")
        if bad:
            print(f"Corrija as {len(bad)} faltantes em {args.evidence} (as métricas de evidência "
                  "ficam subestimadas enquanto elas não existirem no texto).")
        return 0 if not bad else 1

    # ---- Grade de configurações ----
    if args.quick:
        grid = X.make_grid([300, 500], [100])
    else:
        grid = X.make_grid(args.chunk_sizes, args.overlaps)
    if not grid:
        logger.error("Grade vazia: confira --chunk-sizes e --overlaps (overlap < chunk_size).")
        return 1

    out_dir = args.out_dir or ROOT / "data" / "experiments" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("Configurações a avaliar (%d): %s", len(grid), [c.name for c in grid])
    embedder = build_embedder(embedder_kind)

    try:
        full = X.run_experiments(
            root=ROOT,
            questions_file=args.questions,
            evidence_file=args.evidence,
            aliases_file=args.aliases,
            out_dir=out_dir,
            embedder=embedder,
            corpus_dir=corpus_dir,
            grid=grid,
            k_values=args.k_values,
            run_generation=args.generation,
            generation_k=args.k_generation,
            generation_config=args.generation_config,
            generation_sleep=args.sleep,
        )
    except RuntimeError as e:
        logger.error("%s", e)
        return 1

    # ---- Resumo no terminal ----
    by_name = {c["config"]: c for c in full["configs"]}
    print("\n" + "=" * 70)
    print("  RESULTADO (ordenado do melhor para o pior)")
    print("=" * 70)
    for name in full["ranking"]:
        c = by_name[name]
        r1 = c["primary_recall"].get("1", 0) * 100
        rk = c["primary_recall"].get(str(max(args.k_values)), 0) * 100
        star = " <- melhor" if name == full["best"] else ""
        print(f"  {name:<12} chunks={c['n_chunks']:<5} MRR={c['primary_mrr']:.3f} "
              f"R@1={r1:5.1f}% R@{max(args.k_values)}={rk:5.1f}%{star}")
    gen = full.get("generation")
    if gen and gen.get("error"):
        print(f"\n  Geração NÃO executada: {gen['error']}")
    print(f"\n  Relatório: {out_dir / 'report.md'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
