import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

import datasets


POPULATIONS = ["AAA", "BBB", "CCC"]
COLORS = ["#111111", "#222222", "#333333"]


@pytest.fixture
def genomes_dir(tmp_path, monkeypatch):
    """Small fake data in the format of the low-dim-embeddings-review repository."""
    monkeypatch.setattr(datasets, "DATA", tmp_path)
    path = tmp_path / "genomes-data"
    path.mkdir()
    X = np.random.default_rng(0).integers(0, 3, size=(40, 120), dtype=np.int8)
    np.savez(path / "gt_sum_thinned.npz", X=X)
    pops = [POPULATIONS[i % 3] for i in range(len(X))]
    pd.DataFrame({"sample": [f"HG{i:05d}" for i in range(len(X))], "pop": pops, "in phase 3": 1}).to_csv(
        path / "affy_samples.20141118.panel", sep="\t", index=False)
    pd.DataFrame({"pop": POPULATIONS[::-1], "color": COLORS[::-1]}).to_csv(path / "population_colors.csv", index=False)
    return path, X


def test_load_genomes_returns_the_euclidean_distance_matrix(genomes_dir):
    _, X = genomes_dir
    D, y, precomputed = datasets.load_genomes()
    assert precomputed and D.dtype == np.float64 and D.shape == (40, 40)
    np.testing.assert_allclose(D, squareform(pdist(X.astype(np.float64))), rtol=1e-12)
    assert (np.diag(D) == 0).all() and (D == D.T).all()


def test_load_genomes_labels_are_population_colors(genomes_dir):
    _, y, _ = datasets.load_genomes()
    assert list(y) == [COLORS[i % 3] for i in range(40)]


def test_genotype_distances_does_not_depend_on_the_chunk_size(genomes_dir):
    _, X = genomes_dir
    np.testing.assert_array_equal(datasets.genotype_distances(X, chunk_size=7), datasets.genotype_distances(X))


def test_load_genomes_rejects_a_population_without_color(genomes_dir):
    path, _ = genomes_dir
    pd.DataFrame({"pop": POPULATIONS[:2], "color": COLORS[:2]}).to_csv(path / "population_colors.csv", index=False)
    with pytest.raises(ValueError):
        datasets.load_genomes()


def test_load_mnist_uses_the_liac_arff_parser(monkeypatch):
    """Regression guard: sklearn's default "pandas" parser raises a TypeError with pandas >= 3."""
    calls = {}

    def fake_fetch_openml(name, **kwargs):
        calls.update(kwargs, name=name)
        return np.zeros((5, 3)), np.array(list("01234"), dtype=object)

    monkeypatch.setattr(datasets, "fetch_openml", fake_fetch_openml)
    X, y, precomputed = datasets.load_mnist()
    assert calls["name"] == "mnist_784" and calls["parser"] == "liac-arff"
    assert y.dtype.kind == "i" and list(y) == [0, 1, 2, 3, 4] and not precomputed
