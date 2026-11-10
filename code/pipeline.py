"""The three pipeline stages, shared by the main experiments and the ablations.

  run_embeddings   run methods, save the embedding + runtime       (slow, rarely changes)
  run_evaluations  compute metrics for saved embeddings            (fast, changes often)
  collect          join runtimes and metrics into one table

Every stage is incremental: it only does work that is missing or out of date.
"""
import json
import os
import platform
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd

import results_io as rio
from datasets import DATASETS
from methods import METHODS
from metrics import METRICS, EvalContext
from paths import RESULTS


def _log(message):
    print(message, flush=True)


def _unique(jobs, attr):
    seen = {}
    for job in jobs:
        seen.setdefault(getattr(job, attr), job)
    return list(seen.values())


def _by_dataset(jobs):
    groups = defaultdict(list)
    for job in jobs:
        groups[job.dataset].append(job)
    return groups.items()


def write_environment():
    """Record versions and hardware next to the results: runtimes depend on them."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    for name in ["numpy", "scipy", "sklearn", "openTSNE", "umap", "torch"]:
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = None
    if info["torch"]:
        import torch

        info["torch_threads"] = torch.get_num_threads()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(info, indent=1))


# --- stage 1: embeddings ------------------------------------------------------------

def embedding_is_current(job):
    """The saved embedding exists and was computed with the current settings."""
    if not job.emb_path.exists():
        return False
    try:
        return rio.read_embedding_meta(job.emb_path)["kwargs"] == rio.canonical(job.kwargs)
    except Exception:
        return False  # unreadable file: recompute


WARM_UP = True
_warmed = set()


def _warm_up(method, precomputed):
    """Pay the one-off costs of a method (importing its libraries, numba compilation, thread
    start-up) on a tiny synthetic problem, outside the timer, the first time it runs in this
    process. Otherwise the first call of every method would include them (e.g. importing
    torch for the SGD methods, or compiling UMAP and PaCMAP), which can be many seconds.
    Failures are ignored: the real run reports its own errors."""
    if not WARM_UP or (method, precomputed) in _warmed:
        return
    _warmed.add((method, precomputed))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20)) @ rng.normal(size=(20, 20))
    if precomputed:
        from scipy.spatial.distance import cdist

        X = cdist(X, X)
    _log(f"[embed] warm-up (not timed): {method}{' on a distance matrix' if precomputed else ''}")
    try:
        METHODS[method]["fn"](X, precomputed, 0)
    except Exception:
        pass


def run_embeddings(jobs, force=False):
    stats = {"run": 0, "cached": 0, "failed": []}
    for dataset, group in _by_dataset(_unique(jobs, "emb_path")):
        todo = [job for job in group if force or not embedding_is_current(job)]
        stats["cached"] += len(group) - len(todo)
        if not todo:
            continue
        try:
            X, y, precomputed = DATASETS[dataset]["load"]()
        except Exception as error:
            _log(f"[embed] {dataset}: cannot load dataset ({type(error).__name__}: {error})")
            stats["failed"] += [(job, error) for job in todo]
            continue
        for job in todo:
            _warm_up(job.method, precomputed)
            _log(f"[embed] {job.name}, seed {job.emb_seed} ...")
            try:
                start = time.perf_counter()
                out = METHODS[job.method]["fn"](X, precomputed, job.emb_seed, **job.kwargs)
                runtime = time.perf_counter() - start
                Z, extras = out if isinstance(out, tuple) else (out, {})
                rio.save_embedding(job.emb_path, Z, runtime, job.kwargs, extras)
                stats["run"] += 1
                _log(f"[embed]   done in {runtime:.1f} s")
            except Exception as error:
                traceback.print_exc()
                _log(f"[embed]   FAILED: {type(error).__name__}: {error}")
                stats["failed"].append((job, error))
    return stats


# --- stage 2: metrics ----------------------------------------------------------------

def run_evaluations(jobs, metric_names=None, force=()):
    """Compute the metrics that are missing (or listed in force) for every job.
    Metrics computed from an embedding that has since been recomputed count as missing."""
    metric_names = list(metric_names or METRICS)
    stats = {"computed": 0, "up_to_date": 0, "no_embedding": [], "failed": []}
    for dataset, group in _by_dataset(_unique(jobs, "metrics_path")):
        pending = []
        for job in group:
            if not job.emb_path.exists():
                stats["no_embedding"].append(job)
                continue
            meta = rio.read_embedding_meta(job.emb_path)
            have = rio.load_metrics(job.metrics_path, meta["run_id"])
            todo = [name for name in metric_names if name in force or name not in have]
            if todo:
                pending.append((job, meta["run_id"], have, todo))
            else:
                stats["up_to_date"] += 1
        if not pending:
            continue
        try:
            X, y, precomputed = DATASETS[dataset]["load"]()
        except Exception as error:
            _log(f"[evaluate] {dataset}: cannot load dataset ({type(error).__name__}: {error})")
            stats["failed"] += [(job, error) for job, *_ in pending]
            continue
        ctx, ctx_seed = None, None
        for job, run_id, have, todo in sorted(pending, key=lambda p: p[0].seed):
            if ctx_seed != job.seed:  # one context at a time keeps memory bounded
                ctx, ctx_seed = EvalContext(X, y, precomputed, job.seed), job.seed
            _log(f"[evaluate] {job.name}, seed {job.seed}: {', '.join(todo)}")
            Z = rio.load_embedding(job.emb_path)
            for name in todo:
                try:
                    have[name] = METRICS[name](ctx, Z)
                except Exception as error:
                    traceback.print_exc()
                    _log(f"[evaluate]   {name} FAILED: {type(error).__name__}: {error}")
                    stats["failed"].append((job, error))
            rio.save_metrics(job.metrics_path, run_id, have)
            stats["computed"] += 1
    return stats


# --- stage 3: collect ----------------------------------------------------------------

def collect(jobs, metric_names=None):
    """One row per job whose embedding exists. Returns (DataFrame, report), where
    report maps a job name to the seeds with a missing embedding / missing metrics."""
    metric_names = list(metric_names or METRICS)
    rows = []
    report = defaultdict(lambda: {"embedding": [], "metrics": defaultdict(list)})
    for job in jobs:
        if not job.emb_path.exists():
            report[job.name]["embedding"].append(job.seed)
            continue
        meta = rio.read_embedding_meta(job.emb_path)
        have = rio.load_metrics(job.metrics_path, meta["run_id"])
        for name in metric_names:
            if name not in have:
                report[job.name]["metrics"][name].append(job.seed)
        rows.append(
            {
                "dataset": job.dataset,
                "method": job.method,
                "seed": job.seed,
                "emb_seed": job.emb_seed,
                "n": meta["n"],
                "runtime": meta["runtime"],
                **job.tags,
                **{name: have.get(name, np.nan) for name in metric_names},
            }
        )
    return pd.DataFrame(rows), report


def print_report(report, n_jobs):
    if not report:
        _log(f"All {n_jobs} planned runs are complete.")
        return
    _log(f"Incomplete ({n_jobs} planned runs in total):")
    for name, entry in report.items():
        if entry["embedding"]:
            _log(f"  {name}: embedding missing for seeds {entry['embedding']}")
        for metric, seeds in entry["metrics"].items():
            _log(f"  {name}: metric {metric} missing for seeds {seeds}")


def print_summary(stage, stats):
    parts = [f"{k}: {v if not isinstance(v, list) else len(v)}" for k, v in stats.items()]
    _log(f"[{stage}] summary -- " + ", ".join(parts))
    failures = stats.get("failed", [])
    for job, error in failures:
        _log(f"[{stage}]   failed: {job.name}, seed {job.seed}: {type(error).__name__}: {error}")
