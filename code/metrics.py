"""Evaluation metrics.

Every metric is a function  metric(ctx, Z) -> float  registered in METRICS.
ctx is an EvalContext: everything that depends on the dataset and the evaluation
seed but not on the embedding (the random subsample of points, the distances in
the original space, the sampled triplets, the train/test split). It is computed
lazily and shared by all metrics and all methods, so within one (dataset, seed)
every method is evaluated on exactly the same subsample, triplets and split.

Randomness inside metrics comes from ctx.rng(name), which depends only on the
seed and the metric's name, so results do not depend on the order in which
metrics happen to be computed.

To add a metric, write a function and add one line to METRICS.
"""
import warnings
import zlib
from functools import cached_property

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import isotonic_regression
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class EvalContext:
    def __init__(self, X, y, precomputed, seed, subset_size=5000, triplet_reps=10, knn_k=10):
        self.X, self.y, self.precomputed, self.seed = X, y, precomputed, seed
        self.subset_size = subset_size
        self.triplet_reps = triplet_reps
        self.knn_k = knn_k
        self._d2_cache = (None, None)  # (Z, condensed distances of Z[ind])

    def rng(self, name):
        return np.random.default_rng([self.seed, zlib.crc32(name.encode())])

    @cached_property
    def ind(self):
        """Indices of the evaluation subsample (all points if n <= subset_size)."""
        n = self.X.shape[0]
        if n <= self.subset_size:
            return np.arange(n)
        return self.rng("subsample").choice(n, size=self.subset_size, replace=False)

    @cached_property
    def d1v(self):
        """Pairwise distances in the original space within the subsample, condensed
        (upper triangle, row by row; the order of scipy's squareform)."""
        if self.precomputed:
            return squareform(self.X[np.ix_(self.ind, self.ind)], checks=False)
        return pdist(self.X[self.ind])

    @cached_property
    def d1_ranks(self):
        return rankdata(self.d1v)

    @cached_property
    def d1_sorted(self):
        """For stress1_nonmetric: the order of the original distances and their groups of tied
        values. An isotonic regression sorts by the original distances, which never change, so
        this is done once per context and not for every embedding."""
        order = np.argsort(self.d1v, kind="stable")
        sorted_d1 = self.d1v[order]
        starts_group = np.empty(len(sorted_d1), dtype=bool)
        starts_group[0] = True
        np.not_equal(sorted_d1[1:], sorted_d1[:-1], out=starts_group[1:])
        group = np.cumsum(starts_group, dtype=np.int32) - 1
        return order, group, np.bincount(group).astype(float)

    def d2v(self, Z):
        """Pairwise distances within the subsample of the embedding Z (condensed).
        The last result is cached, so consecutive metrics on the same Z share it."""
        if self._d2_cache[0] is not Z:
            self._d2_cache = (Z, pdist(Z[self.ind]))
        return self._d2_cache[1]

    @cached_property
    def triplets(self):
        """For every point and repetition: a random 'close' neighbour (from the nearer
        half of the other points) and a random 'far' one (from the farther half).
        Returns (close, far), each of shape (triplet_reps, m)."""
        m = len(self.ind)
        half = m // 2
        reps = self.triplet_reps
        rng = self.rng("triplets")
        r_close = rng.integers(0, half - 1, size=(reps, m))  # position 1..half-1 in the sorted order
        r_far = rng.integers(0, m - half, size=(reps, m))    # position half..m-1
        d1 = squareform(self.d1v)
        close = np.empty((reps, m), dtype=np.int64)
        far = np.empty((reps, m), dtype=np.int64)
        for start in range(0, m, 500):
            rows = slice(start, min(start + 500, m))
            order = np.argsort(d1[rows], axis=1)
            idx = np.arange(order.shape[0])[None, :]
            close[:, rows] = order[idx, 1 + r_close[:, rows]]  # position 0 is the point itself
            far[:, rows] = order[idx, half + r_far[:, rows]]
        return close, far

    @cached_property
    def knn_high(self):
        """Indices (m, k) of the k nearest neighbours in the original space of each query
        point ctx.ind, searched among ALL n points, the point itself excluded."""
        k = self.knn_k
        if self.precomputed:
            rows = np.array(self.X[self.ind], dtype=float)  # (m, n); a copy
            rows[np.arange(len(self.ind)), self.ind] = np.inf
            return np.argpartition(rows, k, axis=1)[:, :k]
        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(self.X)
        return _without_self(neighbors.kneighbors(self.X[self.ind], return_distance=False), self.ind, k)

    @cached_property
    def split(self):
        """Train/test split (90/10) of all points for the classifiers."""
        return train_test_split(
            np.arange(self.X.shape[0]), test_size=0.1, random_state=self.seed
        )


def _without_self(neighbors, query, k):
    """Drop each query point from its own row of k+1 neighbours (if duplicates pushed it
    out of the row, drop the last neighbour instead), leaving (m, k)."""
    is_self = neighbors == query[:, None]
    is_self[~is_self.any(axis=1), -1] = True
    return neighbors[~is_self].reshape(len(query), k)


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def pearson(ctx, Z):
    """Pearson correlation between pairwise distances before and after embedding."""
    return _corr(ctx.d1v, ctx.d2v(Z))


def spearman(ctx, Z):
    """Spearman correlation between pairwise distances before and after embedding."""
    return _corr(ctx.d1_ranks, rankdata(ctx.d2v(Z)))


def triplet_acc(ctx, Z):
    """Fraction of triplets where the 'far' point is also farther in the embedding."""
    close, far = ctx.triplets
    Zs = Z[ctx.ind]
    anchor = Zs[None, :, :]
    d_close = np.linalg.norm(Zs[close] - anchor, axis=-1)
    d_far = np.linalg.norm(Zs[far] - anchor, axis=-1)
    return float(np.mean(d_far > d_close))


def knn_recall(ctx, Z):
    """Average overlap between the k nearest neighbours (k = 10) of a point in the original
    space and in the embedding, over the query points ctx.ind (all points up to 5000, else a
    random subset); neighbours are searched among all n points."""
    k = ctx.knn_k
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    low = _without_self(neighbors.kneighbors(Z[ctx.ind], return_distance=False), ctx.ind, k)
    overlap = (low[:, :, None] == ctx.knn_high[:, None, :]).any(axis=2).sum(axis=1)
    return float(np.mean(overlap / k))


def knn_acc(ctx, Z):
    """Test accuracy of a 10-nearest-neighbour classifier on the embedding."""
    train, test = ctx.split
    clf = KNeighborsClassifier(n_neighbors=10).fit(Z[train], ctx.y[train])
    return float(clf.score(Z[test], ctx.y[test]))


def lin_acc(ctx, Z):
    """Test accuracy of an (unpenalized) logistic regression on the embedding."""
    train, test = ctx.split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # possible convergence warning
        clf = LogisticRegression(C=np.inf, random_state=ctx.seed).fit(Z[train], ctx.y[train])
    return float(clf.score(Z[test], ctx.y[test]))


# Stress-1 is computed on the evaluation subsample (exact for n <= subset_size),
# for every method, after optimization.
#
# Notation: d = vector of pairwise distances in the original space, d' = the same
# for the embedding Z (same pairs), <.,.> = dot product, |.| = norm.

def stress1_metric(ctx, Z):
    """Metric stress-1 of Z: the normalized stress after rescaling Z optimally.

    The normalized stress of an embedding with scale factor a is
        sqrt( sum (d - a d')^2 / sum d^2 ).
    Here the best a (least squares) is used: a = <d, d'> / <d', d'>.
    (The minimum over a equals sqrt(1 - cos^2), cos being the cosine of the angle
    between the vectors d and d'.)
    Equivalently, this is the minimum over a of Kruskal's stress-1 of the rescaled embedding,
        sqrt( sum (d - a d')^2 / sum (a d')^2 ),    reached at a = |d|^2 / <d, d'>,
    which is the value reported in the paper (both minima equal sqrt(1 - cos^2)).

    Why rescale: methods return embeddings on different scales. SMACOF metric MDS is on
    the scale of the data (digits: rms embedding distance 46 vs 49 for the data), SGD on
    the scale of the normalized distances (rms ~1, digits: 0.94), t-SNE on an arbitrary
    one. Without rescaling, the notebook's stress1 would need normalize_dist=False for
    SMACOF but normalize_dist="mean_square_offdiag" for SGD (each is wrong, ~0.7 or ~1,
    for the other), and neither means anything for e.g. t-SNE.

    Relation to the other stress-1 values: at a minimum of the stress the scale is
    stationary (a = 1) and then |d|^2 = |d'|^2 + |d - d'|^2, so
        sqrt(s^2 / (1 + s^2)) = |d - d'| / |d|,   with s = |d - d'| / |d'|,
    where s is sklearn's `stress_` (normalized_stress=True), i.e. the notebook's
    stress1(adjust=False). So for a converged SMACOF solution this equals the notebook's
    stress1(adjust=True) ("adjusted stress-1"). SGD solutions are only approximately at
    such a minimum, and relative to the normalized D they optimize (against the raw D the
    shortcut gives ~1): digits, shortcut 0.3280 vs exact 0.3279. For t-SNE etc. the
    shortcut is invalid (0.36 or 0.70 vs exact 0.377). Digits, SMACOF: 0.3276 for all.
    """
    d1, d2 = ctx.d1v, ctx.d2v(Z)
    d2 = d2 * (d1 @ d2) / (d2 @ d2)  # best rescaling of the embedding distances
    return float(np.sqrt(np.sum((d1 - d2) ** 2) / np.sum(d1 ** 2)))


def stress1_nonmetric(ctx, Z):
    """Non-metric (Kruskal) stress-1: sqrt( sum (d' - dhat)^2 / sum d'^2 ), where dhat
    is the isotonic regression of d' on d (the "disparities").

    dhat is fitted on the scale of d', so no assumption on the scale of Z is needed.
    This equals the minimum over the scale of Z of stress-1 with f normalized as in the
    non-metric MDS constraint (mean square over the n(n-1)/2 pairs = 1), i.e. the value
    reported in the paper.

    Relation to the notebook's stress1(non_metric=True, normalize_dist="mean_square_offdiag",
    adjust=True): that rescales dhat to rms 1 (a factor c; sklearn does the same, and its
    `stress_` = |d' - c dhat| / |d'| is stress1(adjust=False)) and then applies
    sqrt(s^2 / (1 + s^2)). These two steps undo each other when the scale of Z is
    stationary, |d'|^2 = c |dhat|^2, as at a converged SMACOF solution and approximately
    at an SGD solution (both are on the rms-1 scale), and the notebook version then
    equals this value:
    digits, SMACOF: stress_ 0.2922 -> 0.2805 = this value; SGD: 0.2803 vs 0.2800.
    For embeddings that are not on that scale (t-SNE) the notebook version depends on
    the scale of Z (digits t-SNE: 0.70, 0.71 or 0.64 depending on the scale; this: 0.36).
    """
    # Equal to fitting sklearn's IsotonicRegression(d1 -> d2) (the same PAVA kernel), but without its
    # per-call sorting and validation (12x faster on 12.5M pairs): the pairs are visited in the order of
    # d1 (tied distances merged into their mean, as sklearn does), and the sum of squares is order-free.
    order, group, counts = ctx.d1_sorted
    d2 = ctx.d2v(Z)
    d2_sorted = d2[order]
    group_means = np.bincount(group, weights=d2_sorted) / counts
    disparities = isotonic_regression(group_means, weights=counts, increasing=True).x[group]
    return float(np.sqrt(np.sum((d2_sorted - disparities) ** 2) / np.sum(d2 ** 2)))


METRICS = {
    "pearson": pearson,
    "spearman": spearman,
    "triplet_acc": triplet_acc,
    "knn_recall": knn_recall,
    "knn_acc": knn_acc,
    "lin_acc": lin_acc,
    "stress1_metric": stress1_metric,
    "stress1_nonmetric": stress1_nonmetric,
}
