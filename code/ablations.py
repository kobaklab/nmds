"""Ablations: grids over method settings, run through the same pipeline stages as the
main experiments but kept apart from them (own config, own results folder, own CSVs).

  uv run python code/ablations.py embed                       # all ablations
  uv run python code/ablations.py embed --ablations isotonic_frequency --datasets blobs
  uv run python code/ablations.py evaluate
  uv run python code/ablations.py collect                     # results/ablations/<name>.csv
  uv run python code/ablations.py status

An ablation takes a base method from METHODS and runs it on each point of a grid of
its keyword arguments. The dataset's usual settings for that method (method_kwargs in
datasets.py) apply first and the grid values override them. Stochastic methods get
one embedding per seed; deterministic ones are embedded once, as in the main runs.
Every stage uses the seeds of the entry (default: three seeds; batch_lr, the longest, uses one);
--seeds overrides them for embed and evaluate. An entry with stochastic=True gets one embedding per seed
even if its base method is registered as deterministic (the setting makes it depend on the seed).
To add an ablation, add an entry to ABLATIONS.
"""
import argparse
import itertools

import pipeline
import plan
from datasets import DATASETS
from metrics import METRICS
from paths import RESULTS

ABLATIONS = {
    # How often the isotonic regression is refitted (negative: every |k| epochs,
    # 0: every epoch, positive: every k batches). With 20 epochs on MNIST, refitting
    # every 100 epochs would mean fitting only once, so that point is left out.
    "isotonic_frequency": dict(
        method="sgd_nmds",
        datasets=["blobs", "mnist"],
        grid={"isotonic_fit_how_often": [-100, -10, 0, 10, 1]},
        skip=[("mnist", {"isotonic_fit_how_often": -100})],
    ),
    # Batch size x learning rate of the stochastic metric MDS (200 epochs on Tasic,
    # 20 on MNIST: the defaults for their size).
    "batch_lr": dict(
        method="sgd_mmds",
        datasets=["tasic", "mnist"],
        seeds=[0],  # the longest ablation (24 grid points x 2 datasets): one seed instead of three
        grid={
            "batch_size": [100, 300, 1000, 3000],
            "lr": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        },
    ),
    # Metric MDS after keeping only the first n_pcs principal coordinates.
    "pca_then_mmds": dict(
        method="metric_mds",
        datasets=["genomes"],
        grid={"n_pcs": [5, 10, 15, 20, 30, 40, 50, 100]},
    ),
    # Supplementary figure (code/notebooks/figures_supplement.ipynb): metric MDS from random initializations
    # (three seeds), next to the main run, which starts from classical MDS. metric_mds is registered as
    # deterministic, which holds for that classical start but not for a random one: hence stochastic=True.
    "mmds_random_init": dict(
        method="metric_mds",
        datasets=["blobs"],
        stochastic=True,
        grid={"init": ["random"]},
    ),
}


def ablation_jobs(names=None, datasets=None, seeds=None):
    jobs = []
    for name in names or ABLATIONS:
        spec = ABLATIONS[name]
        method = spec["method"]
        for dataset in spec["datasets"]:
            if datasets and dataset not in datasets:
                continue
            if not plan.is_planned(dataset, method):
                continue
            for values in itertools.product(*spec["grid"].values()):
                point = dict(zip(spec["grid"], values))
                if any(d == dataset and p.items() <= point.items() for d, p in spec.get("skip", [])):
                    continue
                stem = "__".join([method] + [f"{k}={v}" for k, v in point.items()])
                base = RESULTS / "ablations" / name
                for seed in seeds or spec.get("seeds", plan.SEEDS):
                    emb_seed = seed if spec.get("stochastic") else plan.emb_seed_for(method, seed)
                    jobs.append(
                        plan.Job(
                            dataset=dataset,
                            method=method,
                            seed=seed,
                            emb_seed=emb_seed,
                            kwargs=plan.resolve_kwargs(dataset, method, point),
                            emb_path=base / "embeddings" / dataset / f"{stem}_seed{emb_seed}.npz",
                            metrics_path=base / "metrics" / dataset / f"{stem}_seed{seed}.json",
                            tags=point,
                        )
                    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["embed", "evaluate", "collect", "status"])
    parser.add_argument("--ablations", nargs="+", choices=list(ABLATIONS), help="default: all")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), help="default: all")
    parser.add_argument("--seeds", nargs="+", type=int, help=f"default: the seeds of each ablation ({plan.SEEDS}, batch_lr: one)")
    parser.add_argument("--force", action="store_true", help="embed: recompute even if up to date")
    parser.add_argument("--metrics", nargs="+", choices=list(METRICS), help="evaluate: default all")
    parser.add_argument("--force-metrics", nargs="+", choices=list(METRICS), default=[],
                        help="evaluate: recompute these metrics even if already stored")
    args = parser.parse_args()

    if args.stage in ("collect", "status") and (args.datasets or args.seeds):
        parser.error("collect/status always cover all datasets and the configured seeds of the chosen ablations")

    jobs = ablation_jobs(args.ablations, args.datasets, args.seeds)
    if args.stage == "embed":
        pipeline.write_environment()
        pipeline.print_summary("embed", pipeline.run_embeddings(jobs, force=args.force))
    elif args.stage == "evaluate":
        pipeline.print_summary(
            "evaluate", pipeline.run_evaluations(jobs, args.metrics, force=args.force_metrics)
        )
    else:
        for name in args.ablations or ABLATIONS:
            ablation = ablation_jobs([name])
            df, report = pipeline.collect(ablation)
            if args.stage == "collect":
                path = RESULTS / "ablations" / f"{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                print(f"{name}: wrote {len(df)} rows to {path}")
            pipeline.print_report(report, len(ablation))


if __name__ == "__main__":
    main()
