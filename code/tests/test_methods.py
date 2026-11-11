import numpy as np
import pytest
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from methods import METHODS, run_metric_mds

DETERMINISTIC = [name for name, spec in METHODS.items() if spec["deterministic"]]
STOCHASTIC = [name for name, spec in METHODS.items() if not spec["deterministic"]]


@pytest.fixture(scope="module")
def X():
    return make_blobs(n_samples=150, n_features=8, centers=3, random_state=0)[0]


def run(name, X, precomputed, seed, **kwargs):
    out = METHODS[name]["fn"](pairwise_distances(X) if precomputed else X, precomputed, seed, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("name", DETERMINISTIC)
def test_deterministic_methods_ignore_the_seed(name, precomputed, X):
    """The flag in METHODS is what makes embed.py run these once instead of per seed."""
    np.testing.assert_allclose(run(name, X, precomputed, 0), run(name, X, precomputed, 1), atol=1e-8)


@pytest.mark.parametrize("name", STOCHASTIC)
def test_stochastic_methods_depend_on_the_seed_and_are_reproducible(name, X):
    if name.startswith("sgd"):
        pytest.importorskip("torch")
    kwargs = dict(n_epochs=5) if name.startswith("sgd") else {}
    a, b = run(name, X, False, 0, **kwargs), run(name, X, False, 0, **kwargs)
    c = run(name, X, False, 1, **kwargs)
    assert a.shape == (len(X), 2)
    np.testing.assert_allclose(a, b, atol=1e-6)   # same seed, same result
    assert not np.allclose(a, c, atol=1e-4)       # different seed, different result


@pytest.mark.parametrize("name", ["sgd_mmds", "sgd_nmds"])
def test_sgd_methods_return_training_curve_and_accept_settings(name, X):
    pytest.importorskip("torch")
    Z, extras = METHODS[name]["fn"](X, False, 0, n_epochs=4, batch_size=50, lr=0.05)
    assert Z.shape == (len(X), 2)
    assert extras["stress_curve"].shape == (4,)


def test_sgd_works_on_precomputed_distances(X):
    pytest.importorskip("torch")
    Z = run("sgd_nmds", X, True, 0, n_epochs=3)
    assert Z.shape == (len(X), 2) and np.isfinite(Z).all()


def test_classical_mds_precomputed_matches_features(X):
    Zx, Zd = run("classical_mds", X, False, 0), run("classical_mds", X, True, 0)
    np.testing.assert_allclose(pdist(Zx), pdist(Zd), atol=1e-6)


def test_pca_then_metric_mds(X):
    full = run_metric_mds(X, False, 0)
    reduced = run_metric_mds(X, False, 0, n_pcs=3)
    assert full.shape == reduced.shape == (len(X), 2)
    assert not np.allclose(pdist(full), pdist(reduced))
    # same result whether PCA coordinates come from features or from distances
    from_distances = run_metric_mds(pairwise_distances(X), True, 0, n_pcs=3)
    np.testing.assert_allclose(pdist(reduced), pdist(from_distances), rtol=1e-3, atol=1e-3)


def test_umap_is_warmed_up_with_a_matrix_that_reaches_its_approximate_neighbour_search():
    """n >= 4096 switches UMAP from exact to approximate neighbours; pynndescent's breadth-first tree
    building (more numba functions) needs n * 4 * d > 1 MiB on top of that (hybrid_forest.LOCAL_SUBTREE_BYTES)."""
    n, d = METHODS["umap"]["warm_up_shape"]
    assert n >= 4096 and n * 4 * d > 1 << 20


def test_metric_mds_random_init_is_seeded_and_the_default_start_is_unchanged(X):
    from methods import run_metric_mds

    random_0 = run_metric_mds(X, False, 0, init="random")
    np.testing.assert_allclose(random_0, run_metric_mds(X, False, 0, init="random"), atol=1e-8)   # reproducible per seed
    assert not np.allclose(random_0, run_metric_mds(X, False, 1, init="random"), atol=1e-3)         # a different random start
    np.testing.assert_allclose(run_metric_mds(X, False, 0), run_metric_mds(X, False, 1), atol=1e-8)  # classical init: seed-free
