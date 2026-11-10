import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import make_blobs
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import metrics
from metrics import METRICS, EvalContext


@pytest.fixture(scope="module")
def blobs():
    # same toy data as the sklearn-agreement check in the original notebook
    X, y = make_blobs(n_samples=100, random_state=42)
    noise = np.random.default_rng(seed=42).normal(size=(X.shape[0], 1_000))
    return np.concatenate((X, noise), axis=1), y


@pytest.fixture(scope="module")
def sklearn_embeddings(blobs):
    X, _ = blobs
    mmds = MDS(normalized_stress=True, n_init=1, init="classical_mds")
    nmds = MDS(normalized_stress=True, n_init=1, init="classical_mds", metric_mds=False)
    return mmds, mmds.fit_transform(X), nmds, nmds.fit_transform(X)


def notebook_stress1(X, Z, non_metric=False, normalize=False, adjust=False):
    """The stress1() function of the original notebook (printing removed)."""
    d1 = pairwise_distances(X).flatten()
    d2 = pairwise_distances(Z).flatten()
    if non_metric:
        d1 = IsotonicRegression(out_of_bounds="clip").fit_transform(d1, d2).flatten()
    if normalize:
        d1 = d1 / np.sqrt(np.sum(d1 ** 2) / (len(X) ** 2 - len(X)))
    s = np.sqrt(np.mean((d1 - d2) ** 2) / np.mean(d2 ** 2))
    return np.sqrt(s ** 2 / (1 + s ** 2)) if adjust else s


# --- stress-1 -----------------------------------------------------------------

def test_reference_stress_agrees_with_sklearn(blobs, sklearn_embeddings):
    """The notebook's own check: its stress1() reproduces sklearn's stress_."""
    X, _ = blobs
    mmds, Z2, nmds, Z3 = sklearn_embeddings
    assert np.isclose(notebook_stress1(X, Z2), mmds.stress_)
    assert np.isclose(notebook_stress1(X, Z3, non_metric=True, normalize=True), nmds.stress_)


def test_stress_metrics_match_adjusted_stress(blobs, sklearn_embeddings):
    """Our scale-free metrics equal the 'adjusted stress-1' used in the manuscript."""
    X, _ = blobs
    mmds, Z2, nmds, Z3 = sklearn_embeddings
    ctx = EvalContext(X, None, False, seed=0)
    adjusted_metric = np.sqrt(mmds.stress_ ** 2 / (1 + mmds.stress_ ** 2))
    adjusted_nonmetric = notebook_stress1(X, Z3, non_metric=True, normalize=True, adjust=True)
    assert metrics.stress1_metric(ctx, Z2) == pytest.approx(adjusted_metric, rel=1e-4)
    assert metrics.stress1_nonmetric(ctx, Z3) == pytest.approx(adjusted_nonmetric, rel=1e-4)


def test_nonmetric_stress_is_corrected_sklearn_stress(blobs, sklearn_embeddings):
    """sqrt(s^2/(1+s^2)) of sklearn's non-metric stress_ equals Kruskal's stress-1 (docstring claim)."""
    X, _ = blobs
    _, _, nmds, Z3 = sklearn_embeddings
    corrected = np.sqrt(nmds.stress_ ** 2 / (1 + nmds.stress_ ** 2))
    assert metrics.stress1_nonmetric(EvalContext(X, None, False, seed=0), Z3) == pytest.approx(corrected, rel=1e-4)


def test_nonmetric_stress_is_the_notebook_function_without_normalize_and_adjust(blobs, sklearn_embeddings):
    """...and therefore meaningful for any embedding scale (e.g. t-SNE), unlike with normalize/adjust."""
    X, _ = blobs
    _, _, _, Z3 = sklearn_embeddings
    ctx = EvalContext(X, None, False, seed=0)
    for scale in (1.0, 7.3, 0.05):
        expected = notebook_stress1(X, scale * Z3, non_metric=True)
        assert metrics.stress1_nonmetric(ctx, scale * Z3) == pytest.approx(expected, rel=1e-6)


def test_metric_stress_is_stress_after_best_rescaling(blobs, sklearn_embeddings):
    """stress1_metric = min over a of sqrt(sum (d - a d')^2 / sum d^2) = sqrt(1 - cos^2) (docstring claim)."""
    from scipy.optimize import minimize_scalar

    X, _ = blobs
    _, Z2, _, _ = sklearn_embeddings
    d1, d2 = pdist(X), pdist(5.0 * Z2)  # deliberately not at the optimal scale
    brute = minimize_scalar(lambda a: np.sqrt(np.sum((d1 - a * d2) ** 2) / np.sum(d1 ** 2)), bounds=(1e-6, 10), method="bounded", options={"xatol": 1e-12})
    cosine = d1 @ d2 / np.sqrt((d1 @ d1) * (d2 @ d2))
    value = metrics.stress1_metric(EvalContext(X, None, False, seed=0), 5.0 * Z2)
    assert value == pytest.approx(brute.fun, rel=1e-6)
    assert value == pytest.approx(np.sqrt(1 - cosine ** 2), rel=1e-9)


def test_stress_functions_are_the_paper_stress1_minimized_over_the_embedding_scale(blobs, sklearn_embeddings):
    """Paper: Stress-1 = |a d' - f| / |a d'| with f = D (metric) or the normalized isotonic fit (non-metric);
    we report its minimum over the scale a of the embedding, which at a SMACOF solution is a = 1 + s^2
    and equals sqrt(s^2 / (1 + s^2)) for s = sklearn's stress_."""
    from scipy.optimize import minimize_scalar

    X, _ = blobs
    mmds, Z2, nmds, Z3 = sklearn_embeddings
    D = pdist(X)
    ctx = EvalContext(X, None, False, seed=0)

    def minimum_over_scale(f, dp):
        stress = lambda a: np.linalg.norm(a * dp - f) / np.linalg.norm(a * dp)
        result = minimize_scalar(stress, bounds=(1e-3, 1e3), method="bounded", options={"xatol": 1e-12})
        return result.x, result.fun

    # metric: f = D
    a, value = minimum_over_scale(D, pdist(Z2))
    assert value == pytest.approx(metrics.stress1_metric(ctx, Z2), rel=1e-6)
    assert a == pytest.approx(1 + mmds.stress_ ** 2, rel=2e-3)

    # non-metric: f = isotonic fit normalized to mean square 1 over the n(n-1)/2 pairs
    dp = pdist(Z3)
    dhat = IsotonicRegression(out_of_bounds="clip").fit_transform(D, dp)
    f = dhat * np.sqrt(len(D) / (dhat @ dhat))
    a, value = minimum_over_scale(f, dp)
    assert value == pytest.approx(metrics.stress1_nonmetric(ctx, Z3), rel=1e-6)
    assert a == pytest.approx(1 + nmds.stress_ ** 2, rel=2e-3)
    assert np.sqrt(np.sum((dp - f) ** 2) / len(D)) == pytest.approx(value, rel=2e-3)  # sqrt of average stress


@pytest.mark.parametrize("name", ["stress1_metric", "stress1_nonmetric"])
def test_stress_is_scale_invariant(blobs, sklearn_embeddings, name):
    X, _ = blobs
    _, Z2, _, _ = sklearn_embeddings
    ctx = EvalContext(X, None, False, seed=0)
    assert METRICS[name](ctx, Z2) == pytest.approx(METRICS[name](ctx, 7.3 * Z2))


def test_precomputed_equals_features(blobs, sklearn_embeddings):
    X, y = blobs
    _, Z2, _, _ = sklearn_embeddings
    ctx_x = EvalContext(X, y, False, seed=3)
    ctx_d = EvalContext(pairwise_distances(X), y, True, seed=3)
    for name in ["pearson", "spearman", "stress1_metric", "stress1_nonmetric", "triplet_acc"]:
        assert METRICS[name](ctx_x, Z2) == pytest.approx(METRICS[name](ctx_d, Z2), rel=1e-6)


def test_nonmetric_stress_equals_sklearn_isotonic_regression_also_with_many_tied_distances():
    """stress1_nonmetric uses scipy's isotonic regression on a pre-sorted order; sklearn's IsotonicRegression
    is the reference. Integer-valued features give many exactly tied original distances."""
    rng = np.random.default_rng(1)
    X = rng.integers(0, 4, size=(90, 5)).astype(float)
    Z = X[:, :2] + rng.normal(scale=0.8, size=(90, 2))
    d1, d2 = pdist(X), pdist(Z)
    assert len(np.unique(d1)) < len(d1) / 20  # really many ties
    reference = IsotonicRegression(out_of_bounds="clip").fit_transform(d1, d2)
    expected = np.sqrt(np.sum((d2 - reference) ** 2) / np.sum(d2 ** 2))
    assert metrics.stress1_nonmetric(EvalContext(X, None, False, seed=0), Z) == pytest.approx(expected, rel=1e-12)


# --- correlations ---------------------------------------------------------------

def test_pearson_spearman_match_scipy(blobs, sklearn_embeddings):
    X, _ = blobs
    _, Z2, _, _ = sklearn_embeddings
    ctx = EvalContext(X, None, False, seed=0)
    d1, d2 = pdist(X), pdist(Z2)
    assert metrics.pearson(ctx, Z2) == pytest.approx(pearsonr(d1, d2)[0])
    assert metrics.spearman(ctx, Z2) == pytest.approx(spearmanr(d1, d2).statistic)


# --- triplets -------------------------------------------------------------------

def test_triplets_are_valid_and_accuracy_matches_loop(blobs, sklearn_embeddings):
    X, _ = blobs
    _, Z2, _, _ = sklearn_embeddings
    ctx = EvalContext(X, None, False, seed=5, triplet_reps=4)
    close, far = ctx.triplets
    m = len(ctx.ind)
    d1 = pairwise_distances(X[ctx.ind])
    d2 = pairwise_distances(Z2[ctx.ind])
    assert close.shape == far.shape == (4, m)
    # brute force over the same triplets, written like the original double loop
    correct = 0
    for rep in range(4):
        for i in range(m):
            assert close[rep, i] != i and far[rep, i] != i
            rank_close = np.sum(d1[i] < d1[i, close[rep, i]])
            rank_far = np.sum(d1[i] < d1[i, far[rep, i]])
            assert 1 <= rank_close < m // 2 <= rank_far          # nearer half / farther half
            correct += d2[i, far[rep, i]] > d2[i, close[rep, i]]
    assert metrics.triplet_acc(ctx, Z2) == pytest.approx(correct / (4 * m))


def test_triplet_acc_of_perfect_embedding_is_one():
    X = np.random.default_rng(0).normal(size=(200, 5))
    ctx = EvalContext(X, None, False, seed=0)
    assert metrics.triplet_acc(ctx, X) == 1.0


# --- kNN recall -------------------------------------------------------------------

def brute_force_knn_recall(X, Z, query, k):
    """Straight from the definition, with full distance matrices."""
    from sklearn.metrics import pairwise_distances

    d_high, d_low = pairwise_distances(X), pairwise_distances(Z)
    overlaps = []
    for i in query:
        high = set(np.argsort(np.where(np.arange(len(X)) == i, np.inf, d_high[i]))[:k])
        low = set(np.argsort(np.where(np.arange(len(X)) == i, np.inf, d_low[i]))[:k])
        overlaps.append(len(high & low) / k)
    return float(np.mean(overlaps))


@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_recall_matches_brute_force_with_neighbors_searched_among_all_points(precomputed):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 6))
    Z = X[:, :2] + 0.3 * rng.normal(size=(200, 2))
    data = pairwise_distances(X) if precomputed else X
    ctx = EvalContext(data, None, precomputed, seed=1, subset_size=60)   # 60 query points, 200 candidates
    assert len(ctx.ind) == 60 and ctx.knn_high.shape == (60, 10)
    assert metrics.knn_recall(ctx, Z) == pytest.approx(brute_force_knn_recall(X, Z, ctx.ind, 10))


def test_knn_recall_perfect_and_chance_level():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    ctx = EvalContext(X, None, False, seed=0)
    assert metrics.knn_recall(ctx, X) == 1.0
    chance = metrics.knn_recall(ctx, rng.normal(size=(300, 2)))
    assert chance == pytest.approx(10 / 299, abs=0.03)


def test_knn_high_excludes_self_also_with_duplicate_points():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(40, 3)), np.zeros((15, 3))])   # 15 identical points: self can fall out of the k+1 row
    ctx = EvalContext(X, None, False, seed=0, knn_k=10)
    high = ctx.knn_high
    assert high.shape == (len(X), 10)
    assert not (high == ctx.ind[:, None]).any()
    assert all(len(set(row)) == 10 for row in high)


# --- seeding --------------------------------------------------------------------

def test_subsample_used_above_threshold_and_seeded():
    X = np.random.default_rng(0).normal(size=(300, 4))
    a, b = EvalContext(X, None, False, 0, subset_size=100), EvalContext(X, None, False, 0, subset_size=100)
    c = EvalContext(X, None, False, 1, subset_size=100)
    assert len(a.ind) == 100 and len(set(a.ind)) == 100
    assert np.array_equal(a.ind, b.ind)
    assert not np.array_equal(a.ind, c.ind)
    assert np.array_equal(EvalContext(X, None, False, 0, subset_size=300).ind, np.arange(300))


def test_results_do_not_depend_on_metric_order(blobs, sklearn_embeddings):
    X, y = blobs
    _, Z2, _, _ = sklearn_embeddings
    names = list(METRICS)
    forward = {n: METRICS[n](EvalContext(X, y, False, 2), Z2) for n in names}
    ctx = EvalContext(X, y, False, 2)
    backward = {n: METRICS[n](ctx, Z2) for n in reversed(names)}
    assert forward == pytest.approx(backward)


def test_split_matches_original_train_test_split(blobs, sklearn_embeddings):
    X, y = blobs
    _, Z2, _, _ = sklearn_embeddings
    train, test = EvalContext(X, y, False, seed=4).split
    Xtrain, Xtest, ytrain, ytest = train_test_split(Z2, y, random_state=4, test_size=0.1)
    assert np.array_equal(Z2[train], Xtrain) and np.array_equal(Z2[test], Xtest)
    assert np.array_equal(y[train], ytrain) and np.array_equal(y[test], ytest)


def test_all_registered_metrics_run_and_return_floats(blobs, sklearn_embeddings):
    X, y = blobs
    _, Z2, _, _ = sklearn_embeddings
    ctx = EvalContext(X, y, False, 0)
    for name, fn in METRICS.items():
        value = fn(ctx, Z2)
        assert isinstance(value, float) and np.isfinite(value), name
