"""Stage 2: compute metrics for saved embeddings.

  uv run python code/evaluate.py                       # whatever is missing
  uv run python code/evaluate.py --metrics triplet_acc
  uv run python code/evaluate.py --force-metrics triplet_acc   # recompute a changed metric

Results go to results/metrics/<dataset>/<method>_seed<k>.json. Each evaluation seed
uses its own random subsample / triplets / train-test split, so the spread over
seeds includes evaluation noise. Deterministic methods are evaluated for every seed
on their single embedding.
"""
import argparse

import pipeline
import plan
from metrics import METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    plan.add_selection_args(parser)
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), help="default: all")
    parser.add_argument("--force-metrics", nargs="+", choices=list(METRICS), default=[],
                        help="recompute these metrics even if already stored")
    args = parser.parse_args()

    jobs = plan.main_jobs(args.datasets, args.methods, args.seeds)
    stats = pipeline.run_evaluations(jobs, args.metrics, force=args.force_metrics)
    pipeline.print_summary("evaluate", stats)


if __name__ == "__main__":
    main()
