"""Print result tables (mean ± std over seeds) to copy into the manuscript.

  uv run python code/tables.py triplet_acc     # any metric from metrics.py: rows = methods, columns = datasets
  uv run python code/tables.py stress          # stress-1 and runtime, SMACOF vs SGD
  uv run python code/tables.py runtime         # runtime of every method (rows) on every dataset (columns)
  uv run python code/tables.py isotonic        # ablation: how often the isotonic regression is refitted
  uv run python code/tables.py pca             # ablation: PCA before metric MDS on the genomes data
  uv run python code/tables.py summary         # per dataset: every metric and the runtime for every method

Reads results/all_runs.csv and results/ablations/*.csv (see collect.py, ablations.py collect).
Cell markers:  --  not run on purpose (skipped in datasets.py / the ablation config)
               ??  should exist but is missing: run embed.py / evaluate.py / collect.py
               *   best in its column, or within --tol of the best (metric tables only)
"""
import argparse
import sys
from dataclasses import dataclass

import pandas as pd

import plan
from ablations import ABLATIONS, ablation_jobs
from display import DATASET_LABELS, METHOD_LABELS, ROWS, method_for
from methods import METHODS
from metrics import METRICS
from paths import RESULTS

DDOF = 1  # sample standard deviation over seeds; set to 0 for numpy's default np.std
LOWER_IS_BETTER = {"stress1_metric", "stress1_nonmetric"}


@dataclass
class Stat:
    mean: float
    std: float
    n: int


def stat(df, column, **conditions):
    """Mean and std of a column over the rows matching the conditions, or None."""
    rows = df
    for key, value in conditions.items():
        rows = rows[rows[key] == value]
    values = rows[column].dropna()
    if values.empty:
        return None
    std = values.std(ddof=DDOF) if len(values) > 1 else 0.0
    return Stat(float(values.mean()), float(0.0 if pd.isna(std) else std), len(values))


def fmt(s, decimals=2):
    if s is None:
        return "??"
    text = f"{s.mean:.{decimals}f}"
    return text if s.std < 1e-9 else f"{text} ± {s.std:.{decimals}f}"


def fmt_time(s):
    """Seconds (one decimal below 10 s) up to two minutes, minutes above."""
    if s is None:
        return "??"
    if s.mean < 0.1:
        return "<0.1 s"
    unit, scale, decimals = ("s", 1, 1 if s.mean < 10 else 0) if s.mean < 120 else ("m", 60, 1)
    text = f"{s.mean / scale:.{decimals}f}"
    if s.std >= 1e-9:
        text += f" ± {s.std / scale:.{decimals}f}"
    return f"{text} {unit}"


def print_grid(title, header, rows):
    """rows: list of (label, [cell, ...]); aligned plain text."""
    table = [[title] + header] + [[label] + cells for label, cells in rows]
    widths = [max(len(str(r[i])) for r in table) for i in range(len(table[0]))]
    for r in table:
        print("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
    print()


def read_csv(path):
    if not path.exists():
        sys.exit(f"{path} not found. Run the pipeline first (see README).")
    return pd.read_csv(path)


def cell(df, dataset, method, column, formatter=fmt):
    """One table cell for a (dataset, method): '--' if not planned, '??' if missing."""
    if not plan.is_planned(dataset, method):
        return "--", None
    s = stat(df, column, dataset=dataset, method=method)
    return formatter(s), s


# --- tables ---------------------------------------------------------------------------

def metric_table(metric, tol, decimals):
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    cells = {row: [] for row in ROWS}
    stats = {row: [] for row in ROWS}
    for row in ROWS:
        for dataset in datasets:
            text, s = cell(df, dataset, method_for(row, dataset), metric, lambda s: fmt(s, decimals))
            cells[row].append(text)
            stats[row].append(s)
    for j in range(len(datasets)):  # mark the best in each column and everything within tol
        column = [(row, stats[row][j].mean) for row in ROWS if stats[row][j] is not None]
        if not column:
            continue
        pick = min if metric in LOWER_IS_BETTER else max
        best = pick(value for _, value in column)
        for row, value in column:
            if abs(value - best) <= tol + 1e-12:
                cells[row][j] += "*"
    print_grid(metric, [DATASET_LABELS[d] for d in datasets], [(row, cells[row]) for row in ROWS])


def runtime_table():
    """Runtime of every registered method, one row per method (SMACOF and SGD separate)."""
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    methods = [m for m in METHOD_LABELS if m in METHODS] + [m for m in METHODS if m not in METHOD_LABELS]
    rows = [(METHOD_LABELS.get(m, m), [cell(df, d, m, "runtime", fmt_time)[0] for d in datasets]) for m in methods]
    print_grid("Runtime", [DATASET_LABELS[d] for d in datasets], rows)


def stress_table(decimals):
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    rows = [
        ("Metric", "metric_mds", "stress1_metric"),
        ("  with SGD", "sgd_mmds", "stress1_metric"),
        ("Non-metric", "nonmetric_mds", "stress1_nonmetric"),
        ("  with SGD", "sgd_nmds", "stress1_nonmetric"),
    ]
    header = [DATASET_LABELS[d] for d in datasets]
    n = [str(int(df[df.dataset == d].n.iloc[0])) if (df.dataset == d).any() else "??" for d in datasets]
    print_grid("n", header, [("", n)])
    print_grid(
        "Stress-1", header,
        [(label, [cell(df, d, m, col, lambda s: fmt(s, decimals))[0] for d in datasets]) for label, m, col in rows],
    )
    print_grid(
        "Runtime", header,
        [(label, [cell(df, d, m, "runtime", fmt_time)[0] for d in datasets]) for label, m, _ in rows],
    )


def summary_table(decimals):
    """Per dataset: every metric and the runtime, for every method that was planned."""
    df = read_csv(RESULTS / "all_runs.csv")
    columns = list(METRICS)
    for dataset in DATASET_LABELS:
        if not (df.dataset == dataset).any():
            continue
        rows = []
        for method in METHODS:
            if not plan.is_planned(dataset, method):
                continue
            seeds = stat(df, columns[0], dataset=dataset, method=method)
            cells = [str(seeds.n) if seeds else "??"]
            cells += [fmt(stat(df, c, dataset=dataset, method=method), decimals) for c in columns]
            cells.append(fmt_time(stat(df, "runtime", dataset=dataset, method=method)))
            rows.append((method, cells))
        print_grid(f"{dataset} (n={int(df[df.dataset == dataset].n.iloc[0])})", ["seeds"] + columns + ["runtime"], rows)


def planned_points(name):
    """(dataset, grid point) pairs of an ablation that are meant to exist."""
    return {(j.dataset, tuple(j.tags.items())) for j in ablation_jobs([name])}


def ablation_cell(df, planned, dataset, point, column, formatter):
    if (dataset, tuple(point.items())) not in planned:
        return "--"
    return formatter(stat(df, column, dataset=dataset, **point))


def isotonic_table(decimals):
    name = "isotonic_frequency"
    df = read_csv(RESULTS / "ablations" / f"{name}.csv")
    planned = planned_points(name)
    labels = {
        -100: "Once every 100 epochs",
        -10: "Once every 10 epochs",
        0: "Once every epoch (default)",
        10: "Once every 10 batches",
        1: "Every batch",
    }
    datasets = ABLATIONS[name]["datasets"]
    header = [f"{DATASET_LABELS[d]} {what}" for d in datasets for what in ("time", "loss")]
    rows = []
    for value, label in labels.items():
        point = {"isotonic_fit_how_often": value}
        cells = []
        for d in datasets:
            cells.append(ablation_cell(df, planned, d, point, "runtime", fmt_time))
            cells.append(ablation_cell(df, planned, d, point, "stress1_nonmetric", lambda s: fmt(s, decimals)))
        rows.append((label, cells))
    print_grid("Isotonic refit", header, rows)


def pca_table(decimals):
    name = "pca_then_mmds"
    df = read_csv(RESULTS / "ablations" / f"{name}.csv")
    main = read_csv(RESULTS / "all_runs.csv")
    planned = planned_points(name)
    columns = ["triplet_acc", "pearson"]
    rows = []
    for n_pcs in ABLATIONS[name]["grid"]["n_pcs"]:
        rows.append((f"{n_pcs} PCs + metric MDS",
                     [ablation_cell(df, planned, "genomes", {"n_pcs": n_pcs}, c, lambda s: fmt(s, decimals)) for c in columns]))
    for label, method in [("metric MDS, no PCA", "metric_mds"), ("non-metric MDS (SMACOF)", "nonmetric_mds"),
                          ("non-metric MDS (SGD)", "sgd_nmds")]:
        rows.append((label, [fmt(stat(main, c, dataset="genomes", method=method), decimals) for c in columns]))
    print_grid("Genomes", columns, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", choices=["stress", "runtime", "isotonic", "pca", "summary"] + list(METRICS))
    parser.add_argument("--tol", type=float, default=0.02, help="metric tables: also mark values within this of the best")
    parser.add_argument("--decimals", type=int, default=2)
    args = parser.parse_args()
    if args.table == "stress":
        stress_table(args.decimals)
    elif args.table == "runtime":
        runtime_table()
    elif args.table == "isotonic":
        isotonic_table(3)  # the manuscript shows the loss with three decimals
    elif args.table == "pca":
        pca_table(args.decimals)
    elif args.table == "summary":
        summary_table(args.decimals)
    else:
        metric_table(args.table, args.tol, args.decimals)


if __name__ == "__main__":
    main()
