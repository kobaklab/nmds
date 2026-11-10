"""Method registry.

Every method is a function  fn(X, precomputed, seed, **kwargs) -> Z  (or (Z, extras),
where extras is a dict of arrays saved alongside Z, e.g. a training curve).
X is a feature matrix, or an (n, n) distance matrix if precomputed.

deterministic=True means the result does not depend on the seed, so the method is
embedded once per dataset instead of once per seed (tests/test_methods.py checks
that this is true).

To add a comparison method, write a function and add one line to METHODS.
Heavy imports (torch, openTSNE, umap, pacmap) happen inside the functions, so importing
this module is cheap and torch is only needed for the SGD methods.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, ClassicalMDS


def _classical_coords(X, precomputed, n_components=2):
    """Classical MDS. For feature matrices this is PCA (same embedding up to sign,
    and it does not need an n x n distance matrix, which is out of reach for MNIST).
    svd_solver="full" keeps it deterministic."""
    if precomputed:
        return ClassicalMDS(n_components=n_components, metric="precomputed").fit_transform(X)
    return PCA(n_components=n_components, svd_solver="full").fit_transform(X)


def run_classical_mds(X, precomputed, seed):
    return _classical_coords(X, precomputed)


def _sklearn_mds(X, precomputed, metric_mds):
    return MDS(
        normalized_stress=True,
        n_init=1,
        init="classical_mds",
        metric_mds=metric_mds,
        metric="precomputed" if precomputed else "euclidean",
    ).fit_transform(X)


def run_metric_mds(X, precomputed, seed, n_pcs=None):
    """sklearn (SMACOF) metric MDS; with n_pcs, on the first n_pcs classical-MDS/PCA
    coordinates instead of on the raw data."""
    if n_pcs is not None:
        X = _classical_coords(X, precomputed, n_components=n_pcs)
        precomputed = False
    return _sklearn_mds(X, precomputed, metric_mds=True)


def run_nonmetric_mds(X, precomputed, seed):
    return _sklearn_mds(X, precomputed, metric_mds=False)


def run_tsne(X, precomputed, seed):
    from openTSNE import TSNE

    kwargs = dict(metric="precomputed") if precomputed else {}
    return np.asarray(TSNE(random_state=seed, **kwargs).fit(X))


def run_pacmap(X, precomputed, seed):
    """PaCMAP (Wang et al., JMLR 2021). It has no precomputed-distance mode, so for a
    distance matrix it gets the first 100 classical-MDS coordinates, which is what its own
    PCA step (to 100 dimensions) does for high-dimensional data anyway."""
    import pacmap

    if precomputed:
        X = _classical_coords(X, precomputed, n_components=100)
    return pacmap.PaCMAP(n_components=2, random_state=seed).fit_transform(X, init="pca")


def run_umap(X, precomputed, seed):
    from umap import UMAP

    kwargs = dict(metric="precomputed") if precomputed else {}
    return UMAP(random_state=seed, **kwargs).fit_transform(X)


def _run_sgd(X, precomputed, seed, non_metric, **kwargs):
    from stochastic_mds import stochasticMDS

    Z, stress_curve = stochasticMDS(
        X,
        precomputed_distances=precomputed,
        non_metric=non_metric,
        seed=seed,
        verbose=False,
        **kwargs,
    )
    return Z, {"stress_curve": stress_curve}


def run_sgd_mmds(X, precomputed, seed, **kwargs):
    return _run_sgd(X, precomputed, seed, non_metric=False, **kwargs)


def run_sgd_nmds(X, precomputed, seed, **kwargs):
    return _run_sgd(X, precomputed, seed, non_metric=True, **kwargs)


METHODS = {
    "classical_mds": dict(fn=run_classical_mds, deterministic=True),
    "metric_mds": dict(fn=run_metric_mds, deterministic=True),
    "nonmetric_mds": dict(fn=run_nonmetric_mds, deterministic=True),
    "tsne": dict(fn=run_tsne, deterministic=False),
    "umap": dict(fn=run_umap, deterministic=False),
    "pacmap": dict(fn=run_pacmap, deterministic=False),
    "sgd_mmds": dict(fn=run_sgd_mmds, deterministic=False),
    "sgd_nmds": dict(fn=run_sgd_nmds, deterministic=False),
}
