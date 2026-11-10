"""Pipeline logic on a toy registry: skipping, deterministic-once, caching, invalidation,
incremental metrics, stale metrics, collecting. Real methods/datasets are not touched."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

import time

import ablations
import metrics
import pipeline
import plan
import results_io as rio

X_TOY, Y_TOY = make_blobs(n_samples=60, n_features=6, centers=3, random_state=0)
CALLS = []


def det(X, precomputed, seed):
    CALLS.append(("det", seed))
    return PCA(2).fit_transform(X)


def noisy(X, precomputed, seed, scale=0.1):
    CALLS.append(("noisy", seed, scale))
    Z = PCA(2).fit_transform(X)
    return Z + np.random.default_rng(seed).normal(scale=scale, size=Z.shape), {"curve": np.arange(3.0)}


def slow(X, precomputed, seed):
    raise AssertionError("must never run: skipped on every dataset that lists it")


def broken(X, precomputed, seed):
    raise RuntimeError("boom")


@pytest.fixture
def toy(tmp_path, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(pipeline, "WARM_UP", False)   # these tests count method calls; see the warm-up tests below
    datasets = {
        "a": dict(load=lambda: (X_TOY, Y_TOY, False)),
        "b": dict(
            load=lambda: (X_TOY, Y_TOY, False),
            skip=["slow"],
            method_kwargs={"noisy": dict(scale=0.5)},
        ),
    }
    methods = {
        "det": dict(fn=det, deterministic=True),
        "noisy": dict(fn=noisy, deterministic=False),
        "slow": dict(fn=slow, deterministic=False),
    }
    for module in (plan, pipeline, ablations):
        if hasattr(module, "DATASETS"):
            monkeypatch.setattr(module, "DATASETS", datasets)
        if hasattr(module, "METHODS"):
            monkeypatch.setattr(module, "METHODS", methods)
    for module in (plan, pipeline):
        monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(ablations, "RESULTS", tmp_path)
    monkeypatch.setitem(methods, "slow", dict(fn=slow, deterministic=False))
    datasets["a"]["skip"] = ["slow"]
    return datasets, methods


# --- plan ------------------------------------------------------------------------------

def test_plan_skips_and_runs_deterministic_methods_once(toy):
    jobs = plan.main_jobs(seeds=[0, 1, 2])
    assert not any(j.method == "slow" for j in jobs)                    # skipped everywhere
    assert len(jobs) == 2 * 2 * 3                                       # 2 datasets x 2 methods x 3 seeds
    det_jobs = [j for j in jobs if j.method == "det" and j.dataset == "a"]
    assert {j.emb_seed for j in det_jobs} == {0} and len({j.emb_path for j in det_jobs}) == 1
    assert len({j.metrics_path for j in det_jobs}) == 3                 # still evaluated per seed
    noisy_jobs = [j for j in jobs if j.method == "noisy" and j.dataset == "a"]
    assert len({j.emb_path for j in noisy_jobs}) == 3


def test_plan_applies_dataset_specific_method_kwargs(toy):
    kwargs = {(j.dataset, j.method): j.kwargs for j in plan.main_jobs()}
    assert kwargs[("a", "noisy")] == {} and kwargs[("b", "noisy")] == {"scale": 0.5}


# --- embed -----------------------------------------------------------------------------

def test_embeddings_run_once_and_are_cached(toy):
    jobs = plan.main_jobs()
    stats = pipeline.run_embeddings(jobs)
    assert stats["failed"] == [] and stats["run"] == 2 * (1 + 3)        # det once, noisy per seed, per dataset
    assert sorted(c for c in CALLS if c[0] == "det") == [("det", 0), ("det", 0)]
    meta = rio.read_embedding_meta(jobs[0].emb_path)
    assert meta["runtime"] > 0 and meta["n"] == len(X_TOY)
    assert len(rio.load_embedding(next(j.emb_path for j in jobs if j.method == "noisy"), "curve")) == 3

    CALLS.clear()
    stats = pipeline.run_embeddings(jobs)
    assert CALLS == [] and stats["run"] == 0 and stats["cached"] == 8


def test_embedding_is_recomputed_when_settings_change_or_forced(toy):
    datasets, _ = toy
    pipeline.run_embeddings(plan.main_jobs(datasets=["b"], methods=["noisy"]))
    CALLS.clear()
    datasets["b"]["method_kwargs"]["noisy"]["scale"] = 0.9
    pipeline.run_embeddings(plan.main_jobs(datasets=["b"], methods=["noisy"]))
    assert len(CALLS) == 3 and all(c[2] == 0.9 for c in CALLS)
    CALLS.clear()
    pipeline.run_embeddings(plan.main_jobs(datasets=["b"], methods=["noisy"]), force=True)
    assert len(CALLS) == 3


def test_a_failing_method_does_not_stop_the_others(toy):
    _, methods = toy
    methods["broken"] = dict(fn=broken, deterministic=True)
    stats = pipeline.run_embeddings(plan.main_jobs(datasets=["a"], methods=["broken", "det"]))
    assert [j.method for j, _ in stats["failed"]] == ["broken"] and stats["run"] == 1


def test_warm_up_runs_once_per_method_and_is_not_timed(toy, monkeypatch):
    _, methods = toy
    monkeypatch.setattr(pipeline, "WARM_UP", True)
    monkeypatch.setattr(pipeline, "_warmed", set())
    shapes = []

    def slow_first(X, precomputed, seed):
        shapes.append(X.shape)
        if len(shapes) == 1:
            time.sleep(0.4)  # a one-off cost on the first call, like importing torch or compiling with numba
        return PCA(2).fit_transform(X)

    methods["slow_first"] = dict(fn=slow_first, deterministic=False)
    jobs = plan.main_jobs(datasets=["a"], methods=["slow_first"], seeds=[0, 1, 2])
    assert pipeline.run_embeddings(jobs)["run"] == 3
    assert shapes[0] == (300, 20) and shapes[1:] == [X_TOY.shape] * 3   # one warm-up on tiny data, then the three real runs
    assert all(rio.read_embedding_meta(j.emb_path)["runtime"] < 0.3 for j in jobs)


def test_a_method_can_ask_for_a_larger_warm_up_but_distance_matrices_stay_small(toy, monkeypatch):
    _, methods = toy
    monkeypatch.setattr(pipeline, "WARM_UP", True)
    monkeypatch.setattr(pipeline, "_warmed", set())
    shapes = []

    def record(X, precomputed, seed):
        shapes.append((X.shape, precomputed))
        return PCA(2).fit_transform(X)

    methods["big_warm_up"] = dict(fn=record, deterministic=True, warm_up_shape=(500, 7))
    pipeline._warm_up("big_warm_up", False)
    pipeline._warm_up("big_warm_up", True)
    assert shapes == [((500, 7), False), ((300, 300), True)]


def test_a_failing_warm_up_is_ignored(toy, monkeypatch):
    _, methods = toy
    monkeypatch.setattr(pipeline, "WARM_UP", True)
    monkeypatch.setattr(pipeline, "_warmed", set())

    def picky(X, precomputed, seed):
        if len(X) == 300:
            raise RuntimeError("fails on the warm-up problem only")
        return PCA(2).fit_transform(X)

    methods["picky"] = dict(fn=picky, deterministic=True)
    stats = pipeline.run_embeddings(plan.main_jobs(datasets=["a"], methods=["picky"], seeds=[0]))
    assert stats["failed"] == [] and stats["run"] == 1


def test_a_dataset_that_cannot_be_loaded_is_reported_with_its_traceback(toy, capsys):
    datasets, _ = toy

    def broken_loader():
        raise ValueError("cannot parse the file")

    datasets["a"]["load"] = broken_loader
    stats = pipeline.run_embeddings(plan.main_jobs(datasets=["a"], methods=["det"], seeds=[0]))
    captured = capsys.readouterr()
    assert len(stats["failed"]) == 1
    assert "Traceback" in captured.err and "cannot parse the file" in captured.err  # not only the one-line message


# --- evaluate --------------------------------------------------------------------------

def test_every_seed_is_evaluated_even_for_deterministic_methods(toy):
    jobs = plan.main_jobs(datasets=["a"])
    pipeline.run_embeddings(jobs)
    stats = pipeline.run_evaluations(jobs)
    assert stats["failed"] == [] and stats["computed"] == 6
    triplets = []
    for job in (j for j in jobs if j.method == "det"):
        have = rio.load_metrics(job.metrics_path)
        assert set(have) == set(metrics.METRICS)
        triplets.append(have["triplet_acc"])
    assert len(set(triplets)) > 1   # one embedding, different evaluation seeds: evaluation noise shows up


def test_evaluation_is_incremental(toy, monkeypatch):
    jobs = plan.main_jobs(datasets=["a"], methods=["det"])
    pipeline.run_embeddings(jobs)
    pipeline.run_evaluations(jobs)
    assert pipeline.run_evaluations(jobs)["computed"] == 0               # nothing to do

    calls = []
    def new_metric(ctx, Z):
        calls.append(ctx.seed)
        return 1.5
    monkeypatch.setitem(metrics.METRICS, "new_metric", new_metric)
    monkeypatch.setattr(pipeline, "METRICS", metrics.METRICS)
    pipeline.run_evaluations(jobs)
    assert calls == [0, 1, 2]                                            # only the new metric, for every seed
    assert rio.load_metrics(jobs[0].metrics_path)["new_metric"] == 1.5

    pipeline.run_evaluations(jobs, ["new_metric"], force=["new_metric"])
    assert calls == [0, 1, 2, 0, 1, 2]


def test_metrics_of_a_recomputed_embedding_are_stale(toy):
    jobs = plan.main_jobs(datasets=["a"], methods=["noisy"], seeds=[0])
    pipeline.run_embeddings(jobs)
    pipeline.run_evaluations(jobs)
    before = rio.load_metrics(jobs[0].metrics_path)["pearson"]
    pipeline.run_embeddings(jobs, force=True)
    assert pipeline.run_evaluations(jobs)["computed"] == 1               # recomputed, not reused
    assert rio.load_metrics(jobs[0].metrics_path)["pearson"] == pytest.approx(before)  # same seed -> same noise


# --- collect ---------------------------------------------------------------------------

def test_collect_table_and_report(toy):
    jobs = plan.main_jobs()
    pipeline.run_embeddings(jobs)
    pipeline.run_evaluations(jobs)
    df, report = pipeline.collect(jobs)
    assert len(df) == len(jobs) and not report and df["runtime"].gt(0).all()
    assert {"dataset", "method", "seed", "n", "runtime", "pearson", "triplet_acc"} <= set(df.columns)
    # a deterministic method has one runtime, repeated on its three rows
    assert df[(df.method == "det") & (df.dataset == "a")]["runtime"].nunique() == 1

    jobs[-1].emb_path.unlink()
    jobs[0].metrics_path.unlink()
    df, report = pipeline.collect(jobs)
    assert report[jobs[-1].name]["embedding"] == [jobs[-1].seed]
    assert "pearson" in report[jobs[0].name]["metrics"]
    assert df.loc[(df.dataset == jobs[0].dataset) & (df.method == jobs[0].method) & (df.seed == jobs[0].seed), "pearson"].isna().all()


# --- ablations -------------------------------------------------------------------------

def test_real_ablation_grids():
    count = lambda name: len(ablations.ablation_jobs([name]))
    assert count("isotonic_frequency") == (5 + 4) * 3       # MNIST skips "every 100 epochs"
    assert count("batch_lr") == 2 * 24 * 3
    assert count("pca_then_mmds") == 8 * 3
    jobs = ablations.ablation_jobs()
    assert len({j.metrics_path for j in jobs}) == len(jobs)
    pca = [j for j in jobs if j.tags.get("n_pcs") == 10]
    assert len({j.emb_path for j in pca}) == 1               # metric MDS is deterministic: one embedding, 3 evaluations


def test_ablation_grid_overrides_dataset_settings(toy, monkeypatch):
    monkeypatch.setitem(
        ablations.ABLATIONS, "toy",
        dict(method="noisy", datasets=["a", "b"], grid={"scale": [0.2, 0.7]}, skip=[("a", {"scale": 0.7})]),
    )
    jobs = ablations.ablation_jobs(["toy"], seeds=[0])
    assert [(j.dataset, j.kwargs["scale"]) for j in jobs] == [("a", 0.2), ("b", 0.2), ("b", 0.7)]
    stats = pipeline.run_embeddings(jobs)
    assert stats["run"] == 3 and stats["failed"] == []
