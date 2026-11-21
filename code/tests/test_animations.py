"""The snapshot option of stochasticMDS and the SMACOF recorder behind the animations."""
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.manifold import MDS, smacof
from sklearn.metrics import pairwise_distances

from smacof_trajectory import record_smacof


@pytest.fixture(scope="module")
def X():
    return make_blobs(n_samples=150, n_features=8, centers=3, random_state=0)[0]


@pytest.mark.parametrize("non_metric", [False, True])
def test_snapshots_do_not_change_the_optimization(X, non_metric):
    pytest.importorskip("torch")
    from stochastic_mds import stochasticMDS

    kwargs = dict(non_metric=non_metric, n_epochs=4, batch_size=50, verbose=False)
    Z, curve = stochasticMDS(X, **kwargs)
    Zb, curve_b, by_batch = stochasticMDS(X, snapshot_every=1, snapshot_unit="batch", **kwargs)
    Ze, curve_e, by_epoch = stochasticMDS(X, snapshot_every=3, **kwargs)
    for other, other_curve in [(Zb, curve_b), (Ze, curve_e)]:
        np.testing.assert_array_equal(other, Z)
        np.testing.assert_array_equal(other_curve, curve)

    # 3 batches per epoch: the init plus one snapshot per batch, the last one being the result
    assert by_batch["Z"].shape == (13, len(X), 2)
    np.testing.assert_allclose(by_batch["epoch"], np.arange(13) / 3)
    np.testing.assert_array_equal(by_batch["Z"][-1], Z)
    # every 3 epochs out of 4: the init, epoch 3 and (always) the final embedding
    np.testing.assert_array_equal(by_epoch["epoch"], [0, 3, 4])
    np.testing.assert_array_equal(by_epoch["Z"][1], by_batch["Z"][9])
    assert by_epoch["lr"][0] == 1 / (2 * 50) and np.all(np.diff(by_epoch["lr"]) < 0)  # Guttman step, annealed


def test_snapshot_arguments_are_checked(X):
    pytest.importorskip("torch")
    from stochastic_mds import stochasticMDS

    with pytest.raises(ValueError):
        stochasticMDS(X, n_epochs=1, verbose=False, snapshot_every=1, snapshot_unit="batches")
    with pytest.raises(ValueError):
        stochasticMDS(X, n_epochs=1, verbose=False, snapshot_every=0)


@pytest.mark.parametrize("metric_mds", [True, False])
def test_recorded_smacof_is_an_ordinary_sklearn_run(X, metric_mds):
    D = pairwise_distances(X)
    frames, Z, stress = record_smacof(D, metric_mds=metric_mds)
    reference = MDS(metric_mds=metric_mds, metric="precomputed", n_init=1, init="classical_mds",
                    normalized_stress=True)
    np.testing.assert_array_equal(Z, reference.fit_transform(D))
    assert len(frames) == reference.n_iter_ + 1 > 2
    np.testing.assert_array_equal(frames[-1], Z)
    from sklearn.manifold import _mds
    assert _mds.euclidean_distances.__name__ == "euclidean_distances"  # the wrapper is removed again


def test_one_iteration_at_a_time_is_exact_only_for_metric_smacof(X):
    """Why smacof_trajectory records a run instead of chaining one-iteration calls."""
    D = pairwise_distances(X)
    for metric, exact in [(True, True), (False, False)]:
        frames, _, _ = record_smacof(D, metric_mds=metric)
        Y = frames[0]
        for _ in range(len(frames) - 1):
            Y, _ = smacof(D, metric=metric, init=Y, max_iter=1, n_init=1)
        assert np.allclose(Y, frames[-1]) == exact
