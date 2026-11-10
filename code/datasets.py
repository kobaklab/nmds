"""Dataset registry.

Every loader returns (X, y, precomputed): the data matrix, labels, and whether X is
an (n, n) distance matrix instead of a feature matrix.

Per-dataset options:
  skip           methods that are not run on this dataset (e.g. too slow)
  method_kwargs  non-default settings for a method on this dataset
To add a dataset, write a loader and add one entry to DATASETS.
"""
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.datasets import fetch_openml, load_digits, make_blobs

from paths import DATA


def load_blobs():
    X, y = make_blobs(n_samples=5_000, random_state=42)
    noise = np.random.default_rng(seed=42).normal(size=(X.shape[0], 1_000))
    return np.concatenate((X, noise), axis=1), y, False


def load_genomes():
    """Distance matrix of the genomes data. The file may hold the condensed form (n(n-1)/2
    values, scipy's pdist format) or the full n x n matrix, in any float precision; it is
    returned as a full float64 matrix."""
    D = np.load(DATA / "genomes-data/1000-genomes-D.npy")
    if D.ndim == 1:
        D = squareform(D, force="tomatrix")
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"1000-genomes-D.npy must be a condensed or a square distance matrix, got shape {D.shape}")
    # Catches gross mistakes (wrong file, not a distance matrix). The tolerance is relative to the scale of
    # D and loose on purpose: float32 distance routines leave ~sqrt(eps_float32) = 2e-4 noise on the diagonal.
    tolerance = 1e-3 * D.max()
    if not (np.abs(D - D.T).max() <= tolerance and np.abs(np.diag(D)).max() <= tolerance and D.min() >= -tolerance):
        raise ValueError("1000-genomes-D.npy is not a distance matrix (not symmetric, non-zero diagonal or negative values)")
    np.fill_diagonal(D, 0)  # the SGD normalization assumes an exactly zero diagonal
    y = np.load(DATA / "genomes-data/1000-genomes-labels.npy")
    if len(y) != len(D):
        raise ValueError(f"{len(D)} points in the distance matrix but {len(y)} labels")
    return D, y, True


def load_digits_dataset():
    X, y = load_digits(return_X_y=True)
    return X, y, False


def load_tasic():
    X = np.load(DATA / "tasic-data/tasic-pca50.npy")
    y = np.load(DATA / "tasic-data/tasic-colors.npy")
    return X, y, False


def load_mnist():
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, data_home=DATA / "openml"
    )
    return X, y.astype(int), False


# The sklearn (SMACOF) implementations are too slow for the large datasets.
SKLEARN_MDS = ["metric_mds", "nonmetric_mds"]

DATASETS = {
    "blobs": dict(load=load_blobs),
    "genomes": dict(load=load_genomes),
    "digits": dict(load=load_digits_dataset),
    "tasic": dict(
        load=load_tasic,
        skip=SKLEARN_MDS,
        method_kwargs={"sgd_nmds": dict(n_epochs=500)},
    ),
    "mnist": dict(load=load_mnist, skip=SKLEARN_MDS),
}
