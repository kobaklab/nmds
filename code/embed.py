"""Stage 1: compute embeddings for every (dataset, method, seed) and record runtimes.

  uv run python code/embed.py                                  # everything
  uv run python code/embed.py --datasets digits --methods tsne umap
  uv run python code/embed.py --force --methods sgd_nmds       # recompute

Results go to results/embeddings/<dataset>/<method>_seed<k>.npz. Runs whose file
already exists with the same settings are skipped. Methods marked deterministic are
embedded once (seed 0) rather than once per seed.
"""
import argparse

import pipeline
import plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    plan.add_selection_args(parser)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    jobs = plan.main_jobs(args.datasets, args.methods, args.seeds)
    pipeline.write_environment()
    pipeline.print_summary("embed", pipeline.run_embeddings(jobs, force=args.force))


if __name__ == "__main__":
    main()
