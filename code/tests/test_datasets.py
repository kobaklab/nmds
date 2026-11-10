import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import datasets


@pytest.fixture
def genomes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "DATA", tmp_path)
    (tmp_path / "genomes-data").mkdir()
    X = np.random.default_rng(0).normal(size=(40, 5))
    np.save(tmp_path / "genomes-data/1000-genomes-labels.npy", np.arange(40) % 3)
    return tmp_path, X


@pytest.mark.parametrize("form", ["condensed float64", "full float32", "full float64"])
def test_load_genomes_accepts_condensed_and_full_matrices(genomes_dir, form):
    path, X = genomes_dir
    D = pdist(X)
    stored = {"condensed float64": D, "full float32": squareform(D).astype(np.float32), "full float64": squareform(D)}[form]
    np.save(path / "genomes-data/1000-genomes-D.npy", stored)
    loaded, y, precomputed = datasets.load_genomes()
    assert precomputed and loaded.dtype == np.float64 and loaded.shape == (40, 40) and len(y) == 40
    np.testing.assert_allclose(loaded, squareform(D), rtol=1e-6)


@pytest.mark.parametrize("bad", ["not symmetric", "diagonal", "wrong shape"])
def test_load_genomes_rejects_things_that_are_not_distance_matrices(genomes_dir, bad):
    path, X = genomes_dir
    D = squareform(pdist(X))
    if bad == "not symmetric":
        D[0, 1] += 1
    elif bad == "diagonal":
        D = D + 1
    else:
        D = D[:, :30]
    np.save(path / "genomes-data/1000-genomes-D.npy", D)
    with pytest.raises(ValueError):
        datasets.load_genomes()


def test_load_genomes_tolerates_float32_noise_on_the_diagonal(genomes_dir):
    path, X = genomes_dir
    D = squareform(pdist(X)).astype(np.float32)
    np.fill_diagonal(D, 3e-4)  # as left by some float32 distance routines
    np.save(path / "genomes-data/1000-genomes-D.npy", D)
    loaded, _, _ = datasets.load_genomes()
    assert (np.diag(loaded) == 0).all()
