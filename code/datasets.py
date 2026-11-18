"""Dataset registry.

Every loader returns (X, y, precomputed): the data matrix, labels, and whether X is
an (n, n) distance matrix instead of a feature matrix.

Per-dataset options:
  skip           methods that are not run on this dataset (e.g. too slow)
  method_kwargs  non-default settings for a method on this dataset
To add a dataset, write a loader and add one entry to DATASETS.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits, make_blobs

from paths import DATA


def load_blobs():
    X, y = make_blobs(n_samples=5_000, random_state=42)
    noise = np.random.default_rng(seed=42).normal(size=(X.shape[0], 1_000))
    return np.concatenate((X, noise), axis=1), y, False


def genotype_distances(X, chunk_size=5_000):
    """Euclidean distance matrix of the genotype matrix X (small non-negative integers).

    The Gram matrix is accumulated in float32 over column chunks to keep memory low. For the
    1000 Genomes data this is exact: every entry is in {0, 1, 2}, so all partial sums stay
    integers below 2^24, which float32 represents exactly.
    """
    n, p = X.shape
    G = np.zeros((n, n), dtype=np.float32)
    for start in range(0, p, chunk_size):
        block = X[:, start:start + chunk_size].astype(np.float32)
        G += block @ block.T
    G = G.astype(np.float64)
    sq_norms = np.diag(G).copy()
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * G
    np.maximum(D2, 0, out=D2)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0)  # the SGD normalization assumes an exactly zero diagonal
    return D


def load_genomes():
    """1000 Genomes Project data, in the format of the low-dim-embeddings-review repository.

    The genotype matrix is converted into its Euclidean distance matrix, returned as a full
    float64 matrix. The labels are each sample's population color.
    """
    path = DATA / "genomes-data"
    X = np.load(path / "gt_sum_thinned.npz")["X"]
    populations = pd.read_csv(path / "affy_samples.20141118.panel", sep="\t")["pop"].values
    colors = pd.read_csv(path / "population_colors.csv").set_index("pop")["color"]
    if len(populations) != len(X):
        raise ValueError(f"{len(X)} samples in the genotype matrix but {len(populations)} in the panel file")
    missing = sorted(set(populations) - set(colors.index))
    if missing:
        raise ValueError(f"no color for populations {missing} in population_colors.csv")
    y = colors.loc[populations].to_numpy()
    return genotype_distances(X), y, True


def load_digits_dataset():
    X, y = load_digits(return_X_y=True)
    return X, y, False


def load_tasic():
    X = np.load(DATA / "tasic-data/tasic-pca50.npy")
    y = np.load(DATA / "tasic-data/tasic-colors.npy")
    return X, y, False


def load_mnist():
    # parser="liac-arff" on purpose: sklearn's default parser for dense data ("pandas") fails with
    # pandas >= 3 (TypeError in strip_single_quotes: the class labels 0-9 arrive as ints). The
    # liac-arff parser takes ~10 s for MNIST.
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff",
        data_home=DATA / "openml",
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
