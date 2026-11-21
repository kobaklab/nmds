"""Record every iteration of sklearn's SMACOF (metric and non-metric MDS), for animations.

sklearn's MDS only returns the final configuration. Rather than copying its SMACOF loop, this
records the configurations of one ordinary sklearn run: the loop in
sklearn.manifold._mds._smacof_single calls `euclidean_distances(X)` once on the initial
configuration and then once per iteration on the configuration that the Guttman transform
has just produced. Wrapping that one name for the duration of the run captures the whole
trajectory, and the run itself is unchanged.

Why not run SMACOF for one iteration at a time, passing each result as init to the next call?
For metric MDS that would be exact (the state of the algorithm is the configuration alone),
but not for non-metric MDS: sklearn's first iteration uses the (scaled) dissimilarities
themselves instead of their isotonic regression, so every one-iteration call would take a
metric step and the chain would end up at the metric solution. The tests check both claims.

This relies on a private detail of sklearn (tested with 1.8 and 1.9); `record_smacof` checks
that the last recorded configuration equals the returned embedding and raises otherwise.
"""
import contextlib

import numpy as np
from sklearn.manifold import MDS, ClassicalMDS
from sklearn.manifold import _mds


@contextlib.contextmanager
def _recording():
    frames = []
    original = _mds.euclidean_distances

    def recording_euclidean_distances(X, *args, **kwargs):
        frames.append(np.array(X, copy=True))
        return original(X, *args, **kwargs)

    _mds.euclidean_distances = recording_euclidean_distances
    try:
        yield frames
    finally:
        _mds.euclidean_distances = original


def record_smacof(D, metric_mds=True, init="classical_mds", n_components=2, max_iter=300, eps=1e-6,
                  seed=None):
    """Run sklearn's MDS on the (n, n) distance matrix D and return every iteration.

    Same settings as methods.run_metric_mds / run_nonmetric_mds (normalized_stress=True,
    n_init=1, classical-MDS initialization by default). init can also be "random" or an
    (n, n_components) array.

    Returns (frames, embedding, stress): frames has shape (n_iter + 1, n, n_components),
    frames[0] is the initialization and frames[-1] is sklearn's result `embedding`.
    """
    if isinstance(init, str) and init == "classical_mds":
        init = ClassicalMDS(n_components=n_components, metric="precomputed").fit_transform(D)
    mds = MDS(
        n_components=n_components, metric_mds=metric_mds, metric="precomputed", n_init=1,
        init="random" if isinstance(init, str) else "classical_mds",
        max_iter=max_iter, eps=eps, normalized_stress=True, random_state=seed,
    )
    with _recording() as frames:
        Z = mds.fit_transform(D, init=None if isinstance(init, str) else init)
    frames = np.stack(frames)
    if len(frames) != mds.n_iter_ + 1 or not np.allclose(frames[-1], Z):
        raise RuntimeError(
            "Could not record the SMACOF iterations: sklearn's internals have changed "
            f"(recorded {len(frames)} configurations for {mds.n_iter_} iterations)."
        )
    return frames, Z, mds.stress_
