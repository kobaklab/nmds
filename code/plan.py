"""What to run. A Job is one (dataset, method, seed) with its settings and file paths;
embed.py, evaluate.py and collect.py all loop over the same list of jobs, so they
always agree on what should exist.

  skip (datasets.py)        a (dataset, method) pair that is not run at all
  deterministic (methods.py) embedded once, with DETERMINISTIC_SEED, instead of once
                             per seed; it is still *evaluated* for every seed
"""
from dataclasses import dataclass, field
from pathlib import Path

from datasets import DATASETS
from methods import METHODS
from paths import RESULTS

SEEDS = [0, 1, 2]
DETERMINISTIC_SEED = 0


@dataclass
class Job:
    dataset: str
    method: str
    seed: int  # evaluation seed; also the embedding seed unless the method is deterministic
    emb_seed: int
    kwargs: dict  # settings passed to the method
    emb_path: Path
    metrics_path: Path
    tags: dict = field(default_factory=dict)  # extra columns in the collected table

    @property
    def name(self):
        """Human-readable name, without the seed."""
        extra = "".join(f", {k}={v}" for k, v in self.tags.items())
        return f"{self.dataset} / {self.method}{extra}"


def resolve_kwargs(dataset, method, overrides=None):
    """Method settings: the dataset's non-default settings, then any overrides."""
    base = DATASETS[dataset].get("method_kwargs", {}).get(method, {})
    return {**base, **(overrides or {})}


def is_planned(dataset, method):
    return method not in DATASETS[dataset].get("skip", [])


def emb_seed_for(method, seed):
    return DETERMINISTIC_SEED if METHODS[method]["deterministic"] else seed


def main_jobs(datasets=None, methods=None, seeds=None):
    jobs = []
    for dataset in datasets or DATASETS:
        for method in methods or METHODS:
            if not is_planned(dataset, method):
                continue
            for seed in seeds or SEEDS:
                emb_seed = emb_seed_for(method, seed)
                jobs.append(
                    Job(
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        emb_seed=emb_seed,
                        kwargs=resolve_kwargs(dataset, method),
                        emb_path=RESULTS / "embeddings" / dataset / f"{method}_seed{emb_seed}.npz",
                        metrics_path=RESULTS / "metrics" / dataset / f"{method}_seed{seed}.json",
                    )
                )
    return jobs


def add_selection_args(parser):
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), help="default: all")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), help="default: all")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help=f"default: {SEEDS}")
