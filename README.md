# Stochastic MDS and NMDS

Code for the experiments of *Non-metric multidimensional scaling at scale*:
`code/stochastic_mds.py` is the stochastic (mini-batch) metric / non-metric MDS; everything else
runs the comparisons and produces the tables and figures.

* **To use our MDS in your own code**, see the next section: one file, no need for the rest of this repository.
* **To reproduce our experiments**, continue with "Setup for reproducing the experiments".

## Using the stochastic MDS in your own code

`code/stochastic_mds.py` is a single self-contained file (not a package; it imports nothing from the rest of
this repository). Copy it into your project and import the function:

```
pip install numpy scipy "scikit-learn>=1.8" torch     # its only dependencies; CPU-only PyTorch is enough
```

```python
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist, squareform
from stochastic_mds import stochasticMDS

X, y = load_digits(return_X_y=True)                    # data matrix, shape (1797, 64)

Z, stress_curve = stochasticMDS(X)                     # metric MDS
Z, stress_curve = stochasticMDS(X, non_metric=True)    # non-metric MDS: Z has shape (1797, 2)

# from an (n, n) distance matrix instead of a data matrix
D = squareform(pdist(X))
Z, _ = stochasticMDS(D, precomputed_distances=True, non_metric=True)

import matplotlib.pyplot as plt                        # optional
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=3)
```

The three runs above take about 15 seconds on one CPU core (progress is printed every 10 epochs; pass `verbose=False` to silence it).
As a sanity check, the last entry of `stress_curve` on digits is about 0.33 for metric and 0.28 for non-metric MDS.

**Scale.** The distances are normalized in every mini-batch (mean squared off-diagonal value 1), so the defaults
work for any data scale. The returned embedding is on this normalized scale, not on the scale of the data, unlike
the output of scikit-learn's metric MDS.

## Setup for reproducing the experiments

```
uv sync                       # creates .venv (CPU-only PyTorch on Linux/Windows)
```

* Put the genomes and Tasic data into `data/` (see `data/README.md`); MNIST downloads itself.
* The figure notebooks use the matplotlib style in `code/mpl_style.txt`.

## Layout

```
code/
  stochastic_mds.py   the algorithm (needs torch)
  methods.py          registry of embedding methods
  datasets.py         registry of datasets (+ per-dataset skips and settings)
  metrics.py          registry of evaluation metrics
  embed.py            stage 1: run methods, save embeddings and runtimes
  evaluate.py         stage 2: compute metrics for the saved embeddings
  collect.py          stage 3: join everything into results/all_runs.csv, report what is missing
  ablations.py        parameter grids (isotonic frequency, batch size x lr, PCA before MDS, random init), same stages
  tables.py           print mean ± std tables to copy into the manuscript
  display.py          names / row order / which method fills which table row
  plan.py, pipeline.py, results_io.py, paths.py    shared machinery
  notebooks/          figures only: they read saved results, they never compute embeddings
  tests/
results/              embeddings/ metrics/ ablations/ all_runs.csv environment.json
figures/
data/
```

## Running the experiments

```
uv run python code/embed.py                    # all datasets x methods x 3 seeds (slow; incremental)
uv run python code/evaluate.py                 # all metrics (fast)
uv run python code/collect.py                  # results/all_runs.csv + list of anything missing
uv run python code/tables.py stress            # also: runtime (every method), triplet_acc, knn_recall, pearson, ..., isotonic, pca;
                                               # `summary` = all metrics + runtime per method
uv run python code/tables.py stress --latex    # any table (except `summary`) as LaTeX rows for the manuscript, stds in \scriptsize
```

Everything takes `--datasets`, `--methods`, `--seeds` to restrict what runs, e.g.
`uv run python code/embed.py --datasets digits --methods tsne umap`.
Stages are incremental: finished work is skipped, so you can interrupt and resume. `embed.py --force`
recomputes. A crashing run is logged and skipped; `collect.py` tells you what is missing.

Ablations are kept apart from the main runs. Each uses three seeds, except `batch_lr` (the longest), which is configured
with one (`seeds` in its entry in `code/ablations.py`); `--seeds` overrides the configured seeds for `embed` and `evaluate`:

```
uv run python code/ablations.py embed          # also: evaluate, collect, status  (--ablations, --datasets, --seeds)
uv run python code/tables.py isotonic          # and: pca
```

Figures: open `code/notebooks/figures_embeddings.ipynb`, `figures_sgd_params.ipynb` and `figures_supplement.ipynb`
(`uv run jupyter lab`; the last one needs `ablations.py embed/evaluate --ablations mmds_random_init`, see its first cell).
Tests: `uv run pytest`.

## What counts as one run, and where the ± comes from

* Stochastic methods (t-SNE, UMAP, SGD) are embedded once per seed. Methods marked `deterministic` in
  `methods.py` (classical, SMACOF metric / non-metric MDS) are embedded **once**, with seed 0.
  `tests/test_methods.py` checks that the flag is true.
* Every metric is evaluated for **every** seed with that seed: it picks the evaluation subsample (n > 5000),
  the triplets, and the train/test split. So the spread over seeds is optimization noise plus evaluation
  noise; for a deterministic method it is evaluation noise only (and exactly zero for metrics that are exact
  for n ≤ 5000, such as stress-1 on the small datasets).
* Within one (dataset, seed) all methods share the same subsample, triplets and split, so comparisons are paired.
* The std is the sample std (`DDOF = 1` in `tables.py`). Tables show it whenever it is above zero and always for
  stochastic methods, even as `± 0.00` when every seed agrees; a deterministic method (SMACOF, classical MDS) with
  identical values in all seeds has none.

## Metric definitions

* `stress1_metric` = sqrt(1 − cos²(d, d')), i.e. the stress-1 after optimally rescaling the embedding
  distances d' to the original distances d. It equals the "adjusted stress-1" sqrt(s² / (1 + s²)) of a
  converged embedding (the quantity in the old tables) but does not depend on the scale of the embedding,
  which matters for sklearn, t-SNE and UMAP outputs.
* `stress1_nonmetric`: the same with d replaced by its isotonic regression onto d' (Kruskal's stress-1).
* Both are computed on the evaluation subsample, for **all** methods, after optimization. The batch-averaged
  loss of the SGD runs is saved as `stress_curve` in the `.npz` (for training curves), but is not a table value.
* Classical MDS on feature matrices is computed as PCA (identical up to sign, and it avoids the n × n matrix).
* `knn_recall`: for each query point (all points up to 5000, else the evaluation subsample) the overlap between its
  k = 10 nearest neighbours in the original space and in the embedding, averaged over the query points. Neighbours
  are searched among all n points, the point itself is excluded. The high-dimensional neighbours are computed once
  per (dataset, seed) and shared by all methods. (It is at chance level for a dataset like Blobs, where 1000 noise
  dimensions decide who the neighbours are.)
* PaCMAP has no precomputed-distance mode: for a distance matrix (genomes) it gets the first 100 classical-MDS
  coordinates, which is what its own PCA step to 100 dimensions does for high-dimensional data.
* Relation to other stress-1 values: at a converged SMACOF solution both metrics equal the "adjusted" stress-1
  `sqrt(s² / (1 + s²))` of sklearn's `stress_` s (digits: metric 0.3468 → 0.3276, non-metric 0.2922 → 0.2805).
  The value SGD reports for its mini-batches in the last epoch is close but not identical (digits: 0.282 vs 0.280
  after optimization); it is saved as `stress_curve` and not used in the tables. Details in the docstrings in `metrics.py`.

## Extending

* **New method**: write `fn(X, precomputed, seed, **kwargs)` returning `Z` (or `(Z, extras)`), add one line to
  `METHODS` in `methods.py` with `deterministic=True/False`. Then `embed.py --methods new`, `evaluate.py`, `collect.py`.
* **New metric**: write `fn(ctx, Z) -> float`, add one line to `METRICS` in `metrics.py`, run `evaluate.py`: only
  the new metric is computed, for all existing embeddings. After *changing* a metric:
  `evaluate.py --force-metrics name`.
* **New dataset**: loader + one entry in `DATASETS`. Use `skip=[...]` for methods that should not run on it and
  `method_kwargs={method: {...}}` for non-default settings (as for `sgd_nmds` on Tasic). Changing settings
  triggers a recomputation of the affected embeddings automatically.
* **New ablation**: one entry in `ABLATIONS` in `ablations.py` (base method, datasets, grid of kwargs).
* **Table rows**: which method fills "Metric" / "Non-metric" per dataset is in `display.py`.

## Timing notes

Runtime is the wall time of the whole method function (for SGD including its initialization), as run by
`embed.py`, which runs one method at a time. Run it on an otherwise idle machine; `results/environment.json` records
versions and CPU count. UMAP and t-SNE run single-threaded (a seed disables UMAP's parallelism; openTSNE's default is
one job); PaCMAP (numba, faiss) and torch/BLAS in the SGD and SMACOF methods may use all cores, so say in the paper how
many cores were available or pin the thread counts (`NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, ...).

One-off costs (importing libraries such as torch, numba compilation, thread start-up) are not part of the runtime: the
first time a method runs in a process, `embed.py` runs it once on a tiny synthetic problem outside the timer
(`pipeline.WARM_UP`). A method can ask for a larger warm-up with `warm_up_shape` in `methods.py`: UMAP uses 4097 x 128,
because its approximate neighbour search is only reached for n >= 4096 points with n * 4 * d > 1 MiB (with a smaller
warm-up, 3-30 s of numba compilation lands in the first timed UMAP run on blobs, Tasic or MNIST). If a first run still
looks inflated, rerun it with `embed.py --force --methods <name>`: numba caches compiled code on disk, so the rerun is clean.
