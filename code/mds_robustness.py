"""Robustness of metric and non-metric MDS to high-dimensional noise (figure "robustness" and its appendix).

Separate from the rest of this repository's pipeline (no registries, no all_runs.csv): this file runs the
simulations and saves the kNN accuracies to results/robustness/<name>.npz; code/notebooks/figures_robustness.ipynb
makes the figure and the slopes from them.

Simulation. Three clusters in the plane at pairwise distance D, plus d noise dimensions with standard deviation
sigma. Metric and non-metric MDS (scikit-learn, random initialization) and classical MDS (= PCA; deterministic,
exact eigendecomposition) into 2-D, then the leave-one-out 10-NN accuracy of the embedding, averaged over the repetitions. The "breaking" dimension is the smallest d at which that
average is below 0.7; the slope of log(breaking d) against log(D / sigma) is compared with the predictions 2
(metric) and 4 (non-metric).

  uv run python code/mds_robustness.py --jobs 8                # all six settings of the paper, 10 repetitions each
  uv run python code/mds_robustness.py --reps 20 --jobs 8      # more repetitions (seeds 0..19; the saved ones are kept)
  uv run python code/mds_robustness.py ext_uniform             # only some settings (names: keys of CONFIGS)

Settings are in CONFIGS below. A repetition (= its seed) draws the cluster sizes and jitter, one noise realization
and one random initialization of MDS; the first d of the d_max noise columns are used for dimension d, and the same
noise and initialization are used for every D. Results are saved after every repetition, and a rerun only computes
the missing ones.

Why average before thresholding: random-start SMACOF occasionally ends in a bad local minimum (accuracy far below
chance at a single d, between values near 1), and the "first d below 0.7" rule would take that for the collapse.
Averaged over repetitions such dips are harmless (at least three repetitions are needed for that). --n-init 3
(best of three random starts per fit; the paper uses 1) also removes them, at roughly 10x the cost for non-metric MDS.

How it is made cheap. The signal and the noise dimensions are disjoint, so the squared distance between two
points is exactly (squared distance in the plane) + (squared distance in the noise dimensions). The latter comes
from the Gram matrix of the noise, accumulated chunk by chunk (memory does not depend on d_max, which reaches
15 million) and read off at every requested d. MDS then runs on the 99 x 99 (or 297 x 297) distance matrices.
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import linregress
from sklearn.manifold import MDS, ClassicalMDS
from sklearn.neighbors import KNeighborsClassifier

import paths

RESULTS_DIR = paths.RESULTS / "robustness"
CHUNK = 1 << 16  # noise columns generated at a time; the noise is a function of (seed, column), not of this
DEFAULT_REPS = 10
METHODS = ("metric", "non-metric", "classical")  # the first axis of the saved accuracies


@dataclass(frozen=True)
class Config:
    noise: str = "gaussian"  # gaussian, uniform or exponential, scaled to standard deviation sigma
    sigma: float = 1.0
    jitter: bool = False  # move the points away from their cluster centres by N(0, (D/10)^2) in the plane
    n: int = 99
    random_sizes: bool = False  # n ~ Poisson(n), cluster sizes ~ Multinomial(n; 1/3, 1/3, 1/3)
    Ds: tuple = (3.0, 30.0, 10)  # distance between the clusters: (first, last, number of geometric steps)
    extra_Ds: tuple = ()  # more distances, for single curves only (not in the power-law fit)
    dims: tuple = (1, 1_000_000, 30)  # noise dimensionalities, geometric: (first, last, number of steps)
    threshold: float = 0.7  # the accuracy below which the embedding has "collapsed"
    k: int = 10
    n_init: int = 1  # random initializations per MDS fit; the one with the lowest stress is kept


_EXTENDED = dict(jitter=True, random_sizes=True, Ds=(8.0, 64.0, 10), dims=(20, 15_000_000, 30))
CONFIGS = {
    "main": Config(extra_Ds=(10.0,)),  # Fig. robustness: panel a is D = 10, panel b the ten distances in Ds
    "ext_gaussian": Config(**_EXTENDED),  # appendix table, rows 2 to 6
    "ext_uniform": Config(noise="uniform", **_EXTENDED),
    "ext_exponential": Config(noise="exponential", **_EXTENDED),
    "ext_sigma2": Config(sigma=2.0, **_EXTENDED),
    "ext_n297": Config(n=297, **_EXTENDED),
}


def distance_grid(cfg):
    lo, hi, num = cfg.Ds
    return np.concatenate([np.geomspace(lo, hi, num), np.array(cfg.extra_Ds, dtype=float)])


def dims_grid(cfg):
    lo, hi, num = cfg.dims
    return np.geomspace(lo, hi, num, dtype=int)  # as in the original notebook (so the first two can coincide)


# --- one repetition: data -------------------------------------------------------------------------------------

def sample_dataset(cfg, rep):
    """Class labels and the standard Gaussian jitter (scaled by D/10 later)."""
    rng = np.random.default_rng([rep, 0])
    if cfg.random_sizes:
        n = rng.poisson(cfg.n)
        sizes = rng.multinomial(n, [1 / 3] * 3)
    else:
        sizes = [cfg.n // 3, cfg.n // 3, cfg.n - 2 * (cfg.n // 3)]
    y = np.repeat([0, 1, 2], sizes)
    jitter = rng.normal(size=(len(y), 2)) if cfg.jitter else np.zeros((len(y), 2))
    return y, jitter


def signal_squared_distances(D, y, jitter, jitter_scale=10):
    """Squared distances in the plane: clusters at the corners of an equilateral triangle with side D."""
    centres = np.array([[0, 0], [D, 0], [D / 2, D * np.sqrt(3) / 2]])
    points = centres[y] + (D / jitter_scale) * jitter
    sq = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
    return sq


def noise_block(kind, sigma, n, width, rng):
    """n x width noise with standard deviation sigma (mean zero where the distribution allows)."""
    if kind == "gaussian":
        return sigma * rng.normal(size=(n, width))
    if kind == "uniform":
        return sigma * rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, width))
    if kind == "exponential":
        return sigma * rng.exponential(size=(n, width))
    raise ValueError(f"unknown noise type {kind!r}")


def noise_chunk(kind, sigma, n, rep, index):
    """Noise columns [index * CHUNK, (index + 1) * CHUNK) of repetition rep."""
    return noise_block(kind, sigma, n, CHUNK, np.random.default_rng([rep, 1, index]))


def noise_squared_distances(cfg, n, rep, dims):
    """For every d in dims: the squared distances between the points in the first d noise dimensions,
    from the Gram matrix accumulated over the noise chunks. Returns {d: n x n array}."""
    gram = np.zeros((n, n))
    snapshots, done, chunk_index, chunk = {}, 0, None, None
    for target in sorted(set(int(d) for d in dims)):
        while done < target:
            index, offset = divmod(done, CHUNK)
            if index != chunk_index:
                chunk, chunk_index = noise_chunk(cfg.noise, cfg.sigma, n, rep, index), index
            stop = min(target, (index + 1) * CHUNK)
            part = chunk[:, offset : stop - index * CHUNK]
            gram += part @ part.T
            done = stop
        norms = np.diag(gram)
        snapshots[target] = np.maximum(norms[:, None] + norms[None, :] - 2 * gram, 0)
    return snapshots


# --- one repetition: embeddings ---------------------------------------------------------------------------------

def knn_accuracy(Z, y, k=10):
    """Leave-one-out k-NN accuracy: with X=None the points are not their own neighbours."""
    return KNeighborsClassifier(n_neighbors=k).fit(Z, y).score(None, y)


def classical_accuracy(distances, y, k=10):
    """kNN accuracy of classical MDS (= PCA) of a distance matrix: an exact, deterministic eigendecomposition."""
    return knn_accuracy(ClassicalMDS(n_components=2, metric="precomputed").fit_transform(distances), y, k)


def fit_accuracies(distances, y, seed, k=10, n_init=1):
    """kNN accuracy of metric and of non-metric MDS (random initialization) and of classical MDS of a distance
    matrix, in the order of METHODS."""
    accuracies = []
    for metric_mds in (True, False):
        mds = MDS(n_components=2, metric="precomputed", init="random", n_init=n_init, normalized_stress=True,
                  metric_mds=metric_mds, random_state=seed)
        accuracies.append(knn_accuracy(mds.fit_transform(distances), y, k))
    accuracies.append(classical_accuracy(distances, y, k))
    return accuracies


def repetition_matrices(cfg, rep):
    """The class labels of repetition rep, and a generator of its distance matrices: for every distance D (outer loop)
    and every noise dimensionality d (inner loop), the distances between the points."""
    Ds, dims = distance_grid(cfg), dims_grid(cfg)
    y, jitter = sample_dataset(cfg, rep)
    noise_sq = noise_squared_distances(cfg, len(y), rep, dims)
    signal_sq = [signal_squared_distances(D, y, jitter) for D in Ds]

    def matrices():
        for sq_signal in signal_sq:
            for d in dims:
                dist = np.sqrt(sq_signal + noise_sq[int(d)])
                yield (dist + dist.T) / 2  # exactly symmetric, zero diagonal

    return y, matrices()


def run_repetition(cfg, rep, n_jobs=1):
    """Accuracies, array (len(METHODS), number of D, number of d)."""
    y, matrices = repetition_matrices(cfg, rep)
    fits = Parallel(n_jobs=n_jobs)(delayed(fit_accuracies)(dist, y, rep, cfg.k, cfg.n_init) for dist in matrices)
    return np.array(fits).reshape(len(distance_grid(cfg)), len(dims_grid(cfg)), len(METHODS)).transpose(2, 0, 1)


# --- running and saving -------------------------------------------------------------------------------------------

def _path(name):
    return RESULTS_DIR / f"{name}.npz"


def _config_json(cfg):
    return json.dumps(asdict(cfg), sort_keys=True)


def _read(name):
    with np.load(_path(name)) as f:
        config = json.loads(str(f["config"]))
        methods = [str(m) for m in f["methods"]] if "methods" in f.files else ["metric", "non-metric"]  # older files
        return SimpleNamespace(name=name, acc=f["acc"], dims=f["dims"], Ds=f["Ds"], n_fit=int(f["n_fit"]), methods=methods,
                               config=config, sigma=config["sigma"], threshold=config["threshold"], seconds=float(f["seconds"]))


def load(name):
    """Saved results: accuracies acc (len(METHODS), number of D, number of d, repetitions), dims, Ds, n_fit (the first
    n_fit distances are the ones of the power-law fit), the config and the sigma of the noise."""
    result = _read(name)
    if result.methods != list(METHODS):
        raise ValueError(f"{name}: these results were saved before classical MDS was added; "
                         f"rerun: uv run python code/mds_robustness.py {name}")
    return result


def run(name, reps=1, n_jobs=1, force=False):
    cfg = CONFIGS[name]
    done = np.zeros((len(METHODS), len(distance_grid(cfg)), len(dims_grid(cfg)), 0))
    seconds = 0.0
    if _path(name).exists() and not force:
        saved = _read(name)
        if saved.config != json.loads(_config_json(cfg)):
            print(f"[{name}] the saved results are for another config: recomputing")
        elif saved.methods != list(METHODS):
            print(f"[{name}] the saved results predate classical MDS: recomputing")
        else:
            done, seconds = saved.acc, saved.seconds
    if done.shape[-1] >= reps:
        print(f"[{name}] already have {done.shape[-1]} repetition(s)")
    for rep in range(done.shape[-1], reps):
        start = time.perf_counter()
        acc = run_repetition(cfg, rep, n_jobs)
        done = np.concatenate([done, acc[..., None]], axis=-1)
        seconds += time.perf_counter() - start
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_path(name), acc=done, dims=dims_grid(cfg), Ds=distance_grid(cfg), n_fit=cfg.Ds[2],
                 config=_config_json(cfg), seconds=seconds, methods=np.array(METHODS))
        print(f"[{name}] repetition {rep + 1} of {reps} done in {time.perf_counter() - start:.0f} s", flush=True)
    return load(name)


# --- analysis ---------------------------------------------------------------------------------------------------------

def breaking_dims(acc, dims, threshold=0.7):
    """The first d at which the accuracy is below the threshold (nan if it never is). acc: (..., number of d)."""
    below = acc < threshold
    return np.where(below.any(axis=-1), np.asarray(dims, dtype=float)[below.argmax(axis=-1)], np.nan)


def power_law_slope(distances, breaking):
    """Slope of log(breaking dimension) against log(distance); needs two finite values."""
    ok = np.isfinite(breaking)
    if ok.sum() < 2:
        return np.nan
    return linregress(np.log(np.asarray(distances)[ok]), np.log(breaking[ok])).slope


def mean_breaking_dims(result, reps=None):
    """Breaking dimensions, shape (len(METHODS), number of D in the fit): the accuracy is averaged over the repetitions
    (all of them, or the given indices) first, and the threshold is applied to the average."""
    acc = result.acc[:, : result.n_fit]
    if reps is not None:
        acc = acc[..., list(reps)]
    mean = acc.mean(axis=-1)
    return np.array([breaking_dims(mean[m], result.dims, result.threshold) for m in range(len(METHODS))])


def slopes(result, reps=None):
    """Slopes of log(breaking dimension) against log(D / sigma), one per method (METHODS), from the repetition-averaged accuracy."""
    x = result.Ds[: result.n_fit] / result.sigma
    return np.array([power_law_slope(x, breaking) for breaking in mean_breaking_dims(result, reps)])


def summary(result):
    cells = ", ".join(f"{method} {slope:.2f}" for method, slope in zip(result.methods, slopes(result)))
    return (f"[{result.name}] slope of the breaking dimension against D/sigma, accuracy averaged over "
            f"{result.acc.shape[-1]} repetition(s): {cells}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="setting",
                        help=f"settings to run; default: all of them ({', '.join(CONFIGS)})")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help=f"repetitions (seeds 0, 1, ...); default {DEFAULT_REPS}")
    parser.add_argument("--jobs", type=int, default=1, help="parallel MDS fits (joblib); default 1, -1 = all cores")
    parser.add_argument("--n-init", type=int, help="random starts per MDS fit, best kept (default: 1, as in the paper)")
    parser.add_argument("--force", action="store_true", help="recompute even if saved")
    args = parser.parse_args()
    if args.n_init:
        for name in CONFIGS:
            CONFIGS[name] = replace(CONFIGS[name], n_init=args.n_init)  # saved results of another n_init are recomputed
    unknown = [name for name in args.names if name not in CONFIGS]
    if unknown:
        parser.error(f"unknown setting(s) {unknown}; choose from {list(CONFIGS)}")
    for name in dict.fromkeys(args.names or CONFIGS):
        print(summary(run(name, args.reps, args.jobs, args.force)))


if __name__ == "__main__":
    main()
