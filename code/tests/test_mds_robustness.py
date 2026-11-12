import sys

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import mds_robustness as r


def reference_noise(cfg, n, rep, width):
    """The noise matrix n x width, assembled from its chunks."""
    chunks = [r.noise_chunk(cfg.noise, cfg.sigma, n, rep, i) for i in range(-(-width // r.CHUNK))]
    return np.hstack(chunks)[:, :width]


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(r, "CHUNK", 50)  # so that a few hundred dimensions already cross several chunks


@pytest.mark.parametrize("kind", ["gaussian", "uniform", "exponential"])
def test_noise_distances_from_the_gram_matrix_equal_direct_distances(small_chunks, kind):
    cfg = r.Config(noise=kind, sigma=1.5)
    dims = [1, 7, 50, 51, 120, 333, 333]  # inside a chunk, on a boundary, across several chunks, repeated
    snapshots = r.noise_squared_distances(cfg, 12, rep=3, dims=dims)
    noise = reference_noise(cfg, 12, 3, 333)
    for d in set(dims):
        np.testing.assert_allclose(snapshots[d], squareform(pdist(noise[:, :d])) ** 2, atol=1e-9)


def test_the_noise_does_not_depend_on_which_dimensions_are_requested(small_chunks):
    cfg = r.Config()
    a = r.noise_squared_distances(cfg, 10, 0, [40, 200])
    b = r.noise_squared_distances(cfg, 10, 0, [3, 17, 40, 90, 200])
    np.testing.assert_allclose(a[40], b[40], atol=1e-10)
    np.testing.assert_allclose(a[200], b[200], atol=1e-10)
    c = r.noise_squared_distances(cfg, 10, 1, [40])                      # another repetition: other noise
    assert not np.allclose(a[40], c[40])


@pytest.mark.parametrize("kind, mean", [("gaussian", 0.0), ("uniform", 0.0), ("exponential", 2.0)])
def test_noise_has_standard_deviation_sigma(kind, mean):
    block = r.noise_block(kind, 2.0, 50, 20_000, np.random.default_rng(0))
    assert block.std() == pytest.approx(2.0, rel=0.01) and block.mean() == pytest.approx(mean, abs=0.02)


def test_pairwise_noise_distances_concentrate_around_sqrt_2_d_sigma_squared():
    cfg = r.Config(sigma=2.0)
    d = 5000
    sq = r.noise_squared_distances(cfg, 30, 0, [d])[d]
    off_diagonal = sq[~np.eye(30, dtype=bool)]
    assert np.sqrt(off_diagonal).mean() == pytest.approx(np.sqrt(2 * d * 2.0 ** 2), rel=0.01)


def test_clusters_are_equidistant_and_jitter_scales_with_D():
    y = np.repeat([0, 1, 2], 5)
    dist = np.sqrt(r.signal_squared_distances(10.0, y, np.zeros((15, 2))))
    assert np.allclose(dist[y[:, None] != y[None, :]], 10.0) and np.allclose(dist[y[:, None] == y[None, :]], 0.0)
    jitter = np.random.default_rng(0).normal(size=(15, 2))
    moved = np.sqrt(r.signal_squared_distances(20.0, y, jitter))
    within = moved[np.triu_indices(5, 1)]                                   # inside cluster 0: only jitter, scaled by D/10 = 2
    np.testing.assert_allclose(within, 2.0 * pdist(jitter[:5]))


def test_dataset_sampling():
    y, jitter = r.sample_dataset(r.Config(), 0)
    assert np.bincount(y).tolist() == [33, 33, 33] and not jitter.any()
    cfg = r.Config(n=99, random_sizes=True, jitter=True)
    y0, j0 = r.sample_dataset(cfg, 0)
    y0_again, j0_again = r.sample_dataset(cfg, 0)
    assert np.array_equal(y0, y0_again) and np.array_equal(j0, j0_again)       # reproducible per repetition
    sizes = {len(r.sample_dataset(cfg, rep)[0]) for rep in range(20)}
    assert len(sizes) > 5 and all(60 < s < 140 for s in sizes)                  # n ~ Poisson(99)
    assert j0.shape == (len(y0), 2) and j0.any()
    assert not np.array_equal(np.bincount(y0), np.bincount(r.sample_dataset(cfg, 1)[0]))


def test_knn_accuracy_is_leave_one_out():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 2)) + np.repeat([[0, 0], [1, 0], [0, 1]], 20, axis=0)
    y = np.repeat([0, 1, 2], 20)
    D = squareform(pdist(Z))
    np.fill_diagonal(D, np.inf)
    predicted = np.array([np.bincount(y[np.argsort(D[i])[:10]], minlength=3).argmax() for i in range(60)])
    assert r.knn_accuracy(Z, y) == pytest.approx((predicted == y).mean())


def test_breaking_dimension_and_power_law_slope():
    dims = np.array([1, 10, 100, 1000])
    acc = np.array([[1.0, 0.9, 0.6, 0.3], [1.0, 1.0, 0.95, 0.9], [0.5, 0.4, 0.3, 0.2]])
    np.testing.assert_array_equal(r.breaking_dims(acc, dims), [100, np.nan, 1])        # first below 0.7; never; at once
    Ds = np.geomspace(3, 30, 10)
    assert r.power_law_slope(Ds, 7 * Ds ** 2.5) == pytest.approx(2.5)
    broken = 7 * Ds ** 2.5
    broken[[2, 5]] = np.nan                                                          # missing points are left out
    assert r.power_law_slope(Ds, broken) == pytest.approx(2.5)
    assert np.isnan(r.power_law_slope(Ds, np.full(10, np.nan)))


def test_settings_match_the_appendix_table():
    main = r.CONFIGS["main"]
    assert (main.noise, main.sigma, main.jitter, main.n, main.random_sizes) == ("gaussian", 1.0, False, 99, False)
    assert main.Ds == (3.0, 30.0, 10) and main.dims == (1, 1_000_000, 30) and main.threshold == 0.7 and main.k == 10
    rows = {name: (c.noise, c.sigma, c.jitter, c.n, c.random_sizes) for name, c in r.CONFIGS.items() if name != "main"}
    assert rows == {
        "ext_gaussian": ("gaussian", 1.0, True, 99, True),
        "ext_uniform": ("uniform", 1.0, True, 99, True),
        "ext_exponential": ("exponential", 1.0, True, 99, True),
        "ext_sigma2": ("gaussian", 2.0, True, 99, True),
        "ext_n297": ("gaussian", 1.0, True, 297, True),
    }
    assert all(r.CONFIGS[name].Ds == (8.0, 64.0, 10) and r.CONFIGS[name].dims == (20, 15_000_000, 30) for name in rows)
    assert list(r.CONFIGS) == ["main", *rows]


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "RESULTS_DIR", tmp_path / "robustness")
    monkeypatch.setitem(r.CONFIGS, "tiny", r.Config(n=30, Ds=(5.0, 20.0, 3), extra_Ds=(10.0,), dims=(1, 300, 4)))


def test_run_saves_and_does_not_recompute_saved_repetitions(tiny, monkeypatch):
    result = r.run("tiny", reps=2)
    assert result.acc.shape == (3, 4, 4, 2) and result.n_fit == 3                        # 3 methods, 3 + 1 distances, 4 dimensions
    assert result.methods == ["metric", "non-metric", "classical"]
    np.testing.assert_allclose(result.Ds[:3], np.geomspace(5, 20, 3))
    assert result.Ds[3] == 10.0 and result.dims.tolist() == np.geomspace(1, 300, 4, dtype=int).tolist()
    assert (result.acc[:, 2, 0, :] > 0.9).all()                                         # D = 20, one noise dimension: separated

    monkeypatch.setattr(r, "run_repetition", lambda *args, **kwargs: pytest.fail("saved repetitions must not be recomputed"))
    assert r.run("tiny", reps=2).acc.shape[-1] == 2


def test_run_adds_missing_repetitions_and_recomputes_when_the_config_changes(tiny, monkeypatch):
    first = r.run("tiny", reps=1)
    extended = r.run("tiny", reps=3)
    assert extended.acc.shape[-1] == 3
    np.testing.assert_array_equal(extended.acc[..., 0], first.acc[..., 0])         # repetition 0 is reused, not recomputed
    monkeypatch.setitem(r.CONFIGS, "tiny", r.Config(n=30, Ds=(5.0, 20.0, 3), extra_Ds=(10.0,), dims=(1, 200, 4)))
    assert r.run("tiny", reps=1).acc.shape[-1] == 1                                # other dims: starts again


def test_parallel_fits_give_the_same_accuracies(tiny):
    cfg = r.CONFIGS["tiny"]
    serial = r.run_repetition(cfg, 0, n_jobs=1)
    parallel = r.run_repetition(cfg, 0, n_jobs=2)
    np.testing.assert_array_equal(serial, parallel)


def test_slopes_come_from_the_repetition_averaged_accuracy(tiny):
    result = r.run("tiny", reps=3)
    mean_accuracy = result.acc.mean(axis=-1)
    for method in range(len(r.METHODS)):
        expected = r.breaking_dims(mean_accuracy[method, : result.n_fit], result.dims, result.threshold)
        np.testing.assert_array_equal(r.mean_breaking_dims(result)[method], expected)
    assert r.slopes(result).shape == (3,)
    subset = r.mean_breaking_dims(result, reps=[0, 2])                      # a subset of repetitions is averaged on its own
    expected = r.breaking_dims(result.acc[0, : result.n_fit][..., [0, 2]].mean(axis=-1), result.dims, result.threshold)
    np.testing.assert_array_equal(subset[0], expected)
    assert "classical" in r.summary(result) and "3 repetition" in r.summary(result)


def test_the_command_line_runs_all_settings_with_ten_repetitions_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(r, "run", lambda name, reps, n_jobs, force: calls.append((name, reps, n_jobs, force)) or name)
    monkeypatch.setattr(r, "summary", lambda result: f"summary {result}")
    monkeypatch.setattr(sys, "argv", ["mds_robustness.py"])
    r.main()
    assert calls == [(name, 10, 1, False) for name in ["main", "ext_gaussian", "ext_uniform", "ext_exponential", "ext_sigma2", "ext_n297"]]
    calls.clear()
    monkeypatch.setattr(sys, "argv", ["mds_robustness.py", "ext_uniform", "--reps", "3", "--jobs", "2"])
    r.main()
    assert calls == [("ext_uniform", 3, 2, False)]
    monkeypatch.setattr(sys, "argv", ["mds_robustness.py", "no_such_setting"])
    with pytest.raises(SystemExit):
        r.main()


def test_n_init_is_part_of_the_config_and_repairs_a_bad_random_start(tiny, monkeypatch):
    first = r.run("tiny", reps=1)
    assert first.config["n_init"] == 1
    monkeypatch.setitem(r.CONFIGS, "tiny", r.replace(r.CONFIGS["tiny"], n_init=2))
    assert r.run("tiny", reps=1).config["n_init"] == 2                              # another n_init: recomputed, not reused

    # a known bad start: exponential noise, D = 25.4, d = 1331, seed 0 ends at accuracy ~0.24; the best of 3 starts does not
    cfg = r.CONFIGS["ext_exponential"]
    y, jitter = r.sample_dataset(cfg, 0)
    noise_sq = r.noise_squared_distances(cfg, len(y), 0, [1331])[1331]
    dist = np.sqrt(r.signal_squared_distances(r.distance_grid(cfg)[5], y, jitter) + noise_sq)
    dist = (dist + dist.T) / 2
    single = r.fit_accuracies(dist, y, 0, n_init=1)[1]
    if single > 0.7:
        pytest.skip("the known bad random start no longer fails (scikit-learn's random numbers changed)")
    assert r.fit_accuracies(dist, y, 0, n_init=3)[1] > 0.9


def test_classical_mds_is_exact_pca_and_does_not_depend_on_the_seed(small_chunks):
    from sklearn.decomposition import PCA

    cfg = r.Config(n=30)
    y, jitter = r.sample_dataset(cfg, 0)
    D, d = 8.0, 120
    centres = np.array([[0, 0], [D, 0], [D / 2, D * np.sqrt(3) / 2]])
    features = np.concatenate([centres[y] + (D / 10) * jitter, reference_noise(cfg, len(y), 0, d)], axis=1)
    sq = r.signal_squared_distances(D, y, jitter) + r.noise_squared_distances(cfg, len(y), 0, [d])[d]
    dist = np.sqrt(sq)
    dist = (dist + dist.T) / 2
    classical = r.fit_accuracies(dist, y, 0)[2]
    assert classical == r.fit_accuracies(dist, y, 1)[2]                                  # deterministic
    exact_pca = PCA(n_components=2, svd_solver="full").fit_transform(features)           # sklearn's default solver is approximate
    assert classical == pytest.approx(r.knn_accuracy(exact_pca, y))


def test_results_saved_before_classical_mds_are_recomputed_not_mixed(tiny, capsys):
    saved = r.run("tiny", reps=1)
    old = {"acc": saved.acc[:2], "dims": saved.dims, "Ds": saved.Ds, "n_fit": saved.n_fit, "seconds": 1.0,
           "config": r._config_json(r.CONFIGS["tiny"])}                                  # the old format: two methods, no names
    np.savez(r._path("tiny"), **old)
    with pytest.raises(ValueError, match="before classical MDS"):
        r.load("tiny")
    capsys.readouterr()
    fresh = r.run("tiny", reps=1)
    assert "predate classical MDS" in capsys.readouterr().out
    assert fresh.acc.shape[0] == 3 and fresh.methods == list(r.METHODS)


def test_repetition_matrices_come_in_the_order_of_the_saved_array(tiny):
    cfg = r.CONFIGS["tiny"]
    y, matrices = r.repetition_matrices(cfg, 0)
    matrices = list(matrices)
    assert len(matrices) == 4 * 4 and all(m.shape == (30, 30) and np.allclose(m, m.T) and not np.diag(m).any() for m in matrices)
    for first in range(0, 16, 4):                                       # same D (outer loop), more noise dimensions: larger distances
        means = [matrices[first + i].mean() for i in range(4)]
        assert means == sorted(means)
    assert r.classical_accuracy(matrices[3], y) == r.fit_accuracies(matrices[3], y, 0)[2]
