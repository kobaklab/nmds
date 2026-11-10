"""Print result tables (mean ± std over seeds) to copy into the manuscript.

  uv run python code/tables.py triplet_acc     # any metric from metrics.py: rows = methods, columns = datasets
  uv run python code/tables.py stress          # stress-1 and runtime, SMACOF vs SGD
  uv run python code/tables.py runtime         # runtime of every method (rows) on every dataset (columns)
  uv run python code/tables.py isotonic        # ablation: how often the isotonic regression is refitted
  uv run python code/tables.py pca             # ablation: PCA before metric MDS on the genomes data
  uv run python code/tables.py summary         # per dataset: every metric and the runtime for every method

  add --latex  for LaTeX rows to paste into the manuscript (booktabs), the std in a smaller font:
                 metric tables and runtime: \\toprule, header, rows, \\midrule before the t-SNE/UMAP/PaCMAP rows, \\bottomrule
                 stress: \\midrule, the metric rows, \\midrule, the non-metric rows, \\bottomrule; values without std
                 isotonic: \\midrule, the rows, \\bottomrule; times in seconds on both datasets, no stds, --- for skipped
                 pca: the rows only (the header is yours)
                 no LaTeX version of summary

Runtimes are whole seconds, except on tasic and mnist: whole minutes (display.RUNTIME_IN_MINUTES).
The std is shown if it is above zero, and always for stochastic methods (even as 0.00, when all seeds agree);
a deterministic method (SMACOF, classical MDS) with identical values in all seeds has no std.
Reads results/all_runs.csv and results/ablations/*.csv (see collect.py, ablations.py collect).
Cell markers:  --  not run on purpose (skipped in datasets.py / the ablation config)
               ??  should exist but is missing: run embed.py / evaluate.py / collect.py
               *   best in its column, or within --tol of the best (metric tables only; \\textbf in LaTeX)
"""
import argparse
import sys
from dataclasses import dataclass

import pandas as pd

import plan
from ablations import ABLATIONS, ablation_jobs
from display import (
    DATASET_LABELS,
    LATEX_LABELS,
    METHOD_LABELS,
    NEIGHBOUR_METHODS,
    NEIGHBOUR_ROWS,
    ROWS,
    RUNTIME_IN_MINUTES,
    method_for,
)
from methods import METHODS
from metrics import METRICS
from paths import RESULTS

DDOF = 1  # sample standard deviation over seeds; set to 0 for numpy's default np.std
LOWER_IS_BETTER = {"stress1_metric", "stress1_nonmetric"}
SKIPPED = object()  # a (dataset, method) pair that is not run on purpose; shown as --


@dataclass
class Stat:
    mean: float
    std: float
    n: int
    show_std: bool = False  # stochastic method with several seeds: show the std even if every seed agrees


def has_spread(s):
    """A std is shown if it is above zero, or if the method is stochastic and has several seeds."""
    return s.std >= 1e-9 or s.show_std


def stat(df, column, **conditions):
    """Mean and std of a column over the rows matching the conditions, or None."""
    rows = df
    for key, value in conditions.items():
        rows = rows[rows[key] == value]
    values = rows[column].dropna()
    if values.empty:
        return None
    std = values.std(ddof=DDOF) if len(values) > 1 else 0.0
    method = conditions.get("method")
    stochastic = method in METHODS and not METHODS[method]["deterministic"]
    return Stat(float(values.mean()), float(0.0 if pd.isna(std) else std), len(values), show_std=stochastic and len(values) > 1)


# --- formatting of one cell ---------------------------------------------------------------

def _with_std(text, spread, latex):
    if not latex:
        return f"{text} ± {spread}"
    if spread.startswith("0."):  # no leading zero in the small print: .01, not 0.01 (integer runtime stds keep theirs)
        spread = spread[1:]
    return text + rf" {{\scriptsize$\pm${spread}}}"


def fmt(s, decimals=2, latex=False, std=True, bold=False):
    """mean and std. No std if it is exactly zero and the method is deterministic (the metric is exact);
    stochastic methods always show it, even as 0.00. bold: \\textbf on the mean in LaTeX, a trailing * otherwise."""
    if s is None:
        return "??"
    mean = f"{s.mean:.{decimals}f}"
    if bold and latex:
        mean = rf"\textbf{{{mean}}}"
    text = _with_std(mean, f"{s.std:.{decimals}f}", latex) if std and has_spread(s) else mean
    return text + "*" if bold and not latex else text


def fmt_runtime(s, dataset, latex=False, std=True, seconds_only=False):
    """Whole seconds, or whole minutes on the datasets in RUNTIME_IN_MINUTES (unless seconds_only)."""
    if s is None:
        return "??"
    unit, scale = ("m", 60) if dataset in RUNTIME_IN_MINUTES and not seconds_only else ("s", 1)
    text = f"{s.mean / scale:.0f}"
    if std and has_spread(s):
        text = _with_std(text, f"{s.std / scale:.0f}", latex)
    return f"{text} {unit}"


def value(df, dataset, method, column):
    """SKIPPED if the pair is not planned, else the Stat (None if the results are missing)."""
    if not plan.is_planned(dataset, method):
        return SKIPPED
    return stat(df, column, dataset=dataset, method=method)


def render(v, formatter):
    return "--" if v is SKIPPED else formatter(v)


def runtime_cell(df, dataset, method, latex=False, std=True):
    return render(value(df, dataset, method, "runtime"), lambda s: fmt_runtime(s, dataset, latex, std))


# --- output ------------------------------------------------------------------------------------

def print_grid(title, header, rows):
    """rows: list of (label, [cell, ...]); aligned plain text."""
    table = [[title] + header] + [[label] + cells for label, cells in rows]
    widths = [max(len(str(r[i])) for r in table) for i in range(len(table[0]))]
    for r in table:
        print("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths))))
    print()


def _warn_if_missing(rows):
    missing = sum(cells.count("??") for _, cells in rows)
    if missing:
        print(f"warning: {missing} cell(s) are missing and shown as ?? (see collect.py --status)", file=sys.stderr)


def print_latex(blocks, header=None):
    """booktabs rows. With a header: \\toprule, header, \\midrule; without: \\midrule. Then the
    blocks of (label, cells) separated by \\midrule, and \\bottomrule."""
    lines = [r"\toprule", "& " + " & ".join(header) + r" \\"] if header is not None else []
    lines.append(r"\midrule")
    for i, block in enumerate(blocks):
        if i:
            lines.append(r"\midrule")
        lines += [" & ".join([label] + cells) + r" \\" for label, cells in block]
    lines.append(r"\bottomrule")
    print("\n".join(lines))
    _warn_if_missing([row for block in blocks for row in block])


def print_latex_rows(rows):
    """Only the rows: the header of these tables is written by hand."""
    print("\n".join(" & ".join([label] + cells) + r" \\" for label, cells in rows))
    _warn_if_missing(rows)


def latex_blocks(entries, in_second_block):
    """entries: (key, label, cells). Two blocks, split before the first neighbour-embedding row;
    labels in their LaTeX spelling."""
    def tex(entry):
        return LATEX_LABELS.get(entry[1], entry[1]), entry[2]

    first = [tex(e) for e in entries if not in_second_block(e[0])]
    second = [tex(e) for e in entries if in_second_block(e[0])]
    return [block for block in (first, second) if block]


def read_csv(path):
    if not path.exists():
        sys.exit(f"{path} not found. Run the pipeline first (see README).")
    return pd.read_csv(path)


# --- tables ------------------------------------------------------------------------------------

def metric_table(metric, tol, decimals, latex):
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    values = {row: [value(df, d, method_for(row, d), metric) for d in datasets] for row in ROWS}
    pick = min if metric in LOWER_IS_BETTER else max
    best = {}  # per column: the best mean, to mark it and everything within tol
    for j in range(len(datasets)):
        means = [values[row][j].mean for row in ROWS if isinstance(values[row][j], Stat)]
        if means:
            best[j] = pick(means)

    def cell_text(row, j):
        v = values[row][j]
        is_best = isinstance(v, Stat) and abs(v.mean - best[j]) <= tol + 1e-12
        return render(v, lambda s: fmt(s, decimals, latex, bold=is_best))

    entries = [(row, row, [cell_text(row, j) for j in range(len(datasets))]) for row in ROWS]
    header = [DATASET_LABELS[d] for d in datasets]
    if latex:
        print_latex(latex_blocks(entries, lambda row: row in NEIGHBOUR_ROWS), header)
    else:
        print_grid(metric, header, [(label, cells) for _, label, cells in entries])


def runtime_table(latex):
    """Runtime of every registered method, one row per method (SMACOF and SGD separate)."""
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    methods = [m for m in METHOD_LABELS if m in METHODS] + [m for m in METHODS if m not in METHOD_LABELS]
    header = [DATASET_LABELS[d] for d in datasets]
    entries = [(m, METHOD_LABELS.get(m, m), [runtime_cell(df, d, m, latex) for d in datasets]) for m in methods]
    if latex:
        print_latex(latex_blocks(entries, lambda m: m in NEIGHBOUR_METHODS), header)
    else:
        print_grid("Runtime", header, [(label, cells) for _, label, cells in entries])


def stress_table(decimals, latex):
    df = read_csv(RESULTS / "all_runs.csv")
    datasets = list(DATASET_LABELS)
    # (label in plain text, label in LaTeX, method, stress metric)
    rows = [
        ("Metric", "Metric", "metric_mds", "stress1_metric"),
        ("  with SGD", r"\hspace{1em}with SGD", "sgd_mmds", "stress1_metric"),
        ("Non-metric", "Non-metric", "nonmetric_mds", "stress1_nonmetric"),
        ("  with SGD", r"\hspace{1em}with SGD", "sgd_nmds", "stress1_nonmetric"),
    ]

    def stress_cell(d, m, metric, std):
        return render(value(df, d, m, metric), lambda s: fmt(s, decimals, latex, std=std))

    if latex:  # no stds: the caption says how small they are
        table = [
            (tex_label, [stress_cell(d, m, metric, False) for d in datasets] + [runtime_cell(df, d, m, True, False) for d in datasets])
            for _, tex_label, m, metric in rows
        ]
        print_latex([table[:2], table[2:]])
        return
    header = [DATASET_LABELS[d] for d in datasets]
    n = [str(int(df[df.dataset == d].n.iloc[0])) if (df.dataset == d).any() else "??" for d in datasets]
    print_grid("n", header, [("", n)])
    print_grid("Stress-1", header, [(label, [stress_cell(d, m, metric, True) for d in datasets]) for label, _, m, metric in rows])
    print_grid("Runtime", header, [(label, [runtime_cell(df, d, m) for d in datasets]) for label, _, m, _ in rows])


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
            cells.append(fmt_runtime(stat(df, "runtime", dataset=dataset, method=method), dataset))
            rows.append((method, cells))
        print_grid(f"{dataset} (n={int(df[df.dataset == dataset].n.iloc[0])})", ["seeds"] + columns + ["runtime"], rows)


def planned_points(name):
    """(dataset, grid point) pairs of an ablation that are meant to exist."""
    return {(j.dataset, tuple(j.tags.items())) for j in ablation_jobs([name])}


def ablation_cell(df, planned, dataset, point, column, formatter, method):
    if (dataset, tuple(point.items())) not in planned:
        return "--"
    return formatter(stat(df, column, dataset=dataset, method=method, **point))


def isotonic_table(decimals, latex):
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
    method = ABLATIONS[name]["method"]
    header = [f"{DATASET_LABELS[d]} {what}" for d in datasets for what in ("time", "loss")]
    rows = []
    for value_, label in labels.items():
        point = {"isotonic_fit_how_often": value_}
        cells = []
        for d in datasets:  # LaTeX: no stds (the caption gives them); times in seconds on both datasets
            time = lambda s, d=d: fmt_runtime(s, d, latex, std=not latex, seconds_only=True)
            loss = lambda s: fmt(s, decimals, latex, std=not latex)
            cells.append(ablation_cell(df, planned, d, point, "runtime", time, method))
            cells.append(ablation_cell(df, planned, d, point, "stress1_nonmetric", loss, method))
        rows.append((label, ["---" if latex and c == "--" else c for c in cells]))
    print_latex([rows]) if latex else print_grid("Isotonic refit", header, rows)


def pca_table(decimals, latex):
    name = "pca_then_mmds"
    df = read_csv(RESULTS / "ablations" / f"{name}.csv")
    main = read_csv(RESULTS / "all_runs.csv")
    planned = planned_points(name)
    columns = ["triplet_acc", "pearson"]
    rows = []
    for n_pcs in ABLATIONS[name]["grid"]["n_pcs"]:
        rows.append((f"{n_pcs} PCs + metric MDS",
                     [ablation_cell(df, planned, "genomes", {"n_pcs": n_pcs}, c, lambda s: fmt(s, decimals, latex), ABLATIONS[name]["method"])
                      for c in columns]))
    for label, method in [("metric MDS, no PCA", "metric_mds"), ("non-metric MDS (SMACOF)", "nonmetric_mds"),
                          ("non-metric MDS (SGD)", "sgd_nmds")]:
        rows.append((label, [fmt(stat(main, c, dataset="genomes", method=method), decimals, latex) for c in columns]))
    print_latex_rows(rows) if latex else print_grid("Genomes", columns, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", choices=["stress", "runtime", "isotonic", "pca", "summary"] + list(METRICS))
    parser.add_argument("--tol", type=float, default=0.02, help="metric tables: also mark values within this of the best")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--latex", action="store_true", help="LaTeX rows to paste into the manuscript")
    args = parser.parse_args()
    if args.latex and args.table == "summary":
        parser.error("--latex is not available for the summary table")
    if args.table == "stress":
        stress_table(args.decimals, args.latex)
    elif args.table == "runtime":
        runtime_table(args.latex)
    elif args.table == "isotonic":
        isotonic_table(3, args.latex)  # the manuscript shows the loss with three decimals
    elif args.table == "pca":
        pca_table(args.decimals, args.latex)
    elif args.table == "summary":
        summary_table(args.decimals)
    else:
        metric_table(args.table, args.tol, args.decimals, args.latex)


if __name__ == "__main__":
    main()
