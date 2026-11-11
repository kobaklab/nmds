import sys

import pandas as pd
import pytest

import plan
import tables
from metrics import METRICS


def build_csv(tmp_path, overrides, drop=()):
    """An all_runs.csv with a row for every planned (dataset, method, seed): runtime 10 s and every
    metric 0.5, except where overrides[(dataset, method)] = {column: [value per seed]} says otherwise."""
    rows = []
    for job in plan.main_jobs():
        if (job.dataset, job.method) in drop:
            continue
        row = {"dataset": job.dataset, "method": job.method, "seed": job.seed, "emb_seed": job.emb_seed,
               "n": 100, "runtime": 10.0, **{name: 0.5 for name in METRICS}}
        for column, values in overrides.get((job.dataset, job.method), {}).items():
            row[column] = values[job.seed]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "all_runs.csv", index=False)


def run_tables(monkeypatch, capsys, tmp_path, *argv):
    monkeypatch.setattr(tables, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["tables.py", *argv])
    tables.main()
    captured = capsys.readouterr()
    return captured.out, captured.err


def spread(mean, delta):
    return [mean - delta, mean, mean + delta]


def const(value):
    return [value] * 3


# the numbers of the stress table in the manuscript: stress-1 and runtime in seconds
STRESS = {
    # (dataset, method): (stress-1, seconds); SMACOF is deterministic (same value for every seed)
    ("blobs", "metric_mds"): (0.40, 99), ("genomes", "metric_mds"): (0.41, 72), ("digits", "metric_mds"): (0.33, 9),
    ("blobs", "sgd_mmds"): (0.40, 73), ("genomes", "sgd_mmds"): (0.41, 15), ("digits", "sgd_mmds"): (0.33, 4),
    ("tasic", "sgd_mmds"): (0.25, 4 * 60), ("mnist", "sgd_mmds"): (0.36, 14 * 60),
    ("blobs", "nonmetric_mds"): (0.25, 892), ("genomes", "nonmetric_mds"): (0.13, 204), ("digits", "nonmetric_mds"): (0.29, 131),
    ("blobs", "sgd_nmds"): (0.25, 73), ("genomes", "sgd_nmds"): (0.13, 20), ("digits", "sgd_nmds"): (0.29, 6),
    ("tasic", "sgd_nmds"): (0.07, 37 * 60), ("mnist", "sgd_nmds"): (0.29, 17 * 60),
}


def stress_overrides():
    overrides = {}
    for (dataset, method), (stress, seconds) in STRESS.items():
        column = "stress1_metric" if "mmds" in method or method == "metric_mds" else "stress1_nonmetric"
        deterministic = method in ("metric_mds", "nonmetric_mds")
        overrides[(dataset, method)] = {
            column: const(stress) if deterministic else spread(stress, 0.01),
            "runtime": const(seconds) if deterministic else spread(seconds, 10 if seconds > 600 else 2),
        }
    return overrides


def test_stress_latex_is_exactly_the_manuscript_format_without_stds(tmp_path, monkeypatch, capsys):
    build_csv(tmp_path, stress_overrides())
    out, err = run_tables(monkeypatch, capsys, tmp_path, "stress", "--latex")
    assert out.splitlines() == [
        r"\midrule",
        r"Metric & 0.40 & 0.41 & 0.33 & -- & -- & 99 s & 72 s & 9 s & -- & -- \\",
        r"\hspace{1em}with SGD & 0.40 & 0.41 & 0.33 & 0.25 & 0.36 & 73 s & 15 s & 4 s & 4 m & 14 m \\",
        r"\midrule",
        r"Non-metric & 0.25 & 0.13 & 0.29 & -- & -- & 892 s & 204 s & 131 s & -- & -- \\",
        r"\hspace{1em}with SGD & 0.25 & 0.13 & 0.29 & 0.07 & 0.29 & 73 s & 20 s & 6 s & 37 m & 17 m \\",
        r"\bottomrule",
    ]
    assert err == ""


def test_stress_without_latex_has_stds_and_the_same_runtime_rounding(tmp_path, monkeypatch, capsys):
    build_csv(tmp_path, stress_overrides())
    out, _ = run_tables(monkeypatch, capsys, tmp_path, "stress")
    runtime_block = out[out.index("Runtime"):]
    assert "73 ± 2 s" in runtime_block          # whole seconds, std rounded the same way
    assert "9 s" in runtime_block and "9 ±" not in runtime_block   # deterministic: no std
    assert "4 ± 0 m" in runtime_block           # tasic: whole minutes (std 10 s is 0 min)
    assert "37 ± 0 m" in runtime_block and "--" in runtime_block
    assert "0.40 ± 0.01" in out                 # stress-1 keeps its std outside LaTeX


def test_metric_table_latex_layout_bold_and_small_stds(tmp_path, monkeypatch, capsys):
    digits = {"classical_mds": const(0.80), "metric_mds": const(0.84), "nonmetric_mds": const(0.86),
              "tsne": spread(0.74, 0.01), "umap": spread(0.68, 0.01), "pacmap": spread(0.70, 0.01)}
    build_csv(tmp_path, {("digits", m): {"triplet_acc": v} for m, v in digits.items()})
    out, _ = run_tables(monkeypatch, capsys, tmp_path, "triplet_acc", "--latex")
    lines = out.splitlines()
    assert lines[:3] == [r"\toprule", r"& Blobs & Gen. & Digits & Tasic & MNIST \\", r"\midrule"]
    assert lines[-1] == r"\bottomrule"
    assert lines.count(r"\midrule") == 2 and lines[6] == r"\midrule"          # after Classical/Metric/Non-metric
    assert [l.split(" & ")[0] for l in lines[3:6] + lines[7:10]] == ["Classical", "Metric", "Non-metric", "$t$-SNE", "UMAP", "PaCMAP"]
    digits_cells = {l.split(" & ")[0]: l.split(" & ")[3] for l in lines[3:6] + lines[7:10]}
    assert digits_cells["Non-metric"] == r"\textbf{0.86}"
    assert digits_cells["Metric"] == r"\textbf{0.84}"                          # within 0.02 of the best
    assert digits_cells["Classical"] == "0.80"
    assert digits_cells["UMAP"] == r"0.68 {\scriptsize$\pm$.01}"               # std in smaller font, no leading zero, not bold
    assert "*" not in out
    plain, _ = run_tables(monkeypatch, capsys, tmp_path, "triplet_acc")
    assert "0.68 ± 0.01" in plain                                              # outside LaTeX the std keeps its zero


def test_runtime_table_units_rounding_and_blocks(tmp_path, monkeypatch, capsys):
    overrides = {
        ("digits", "classical_mds"): {"runtime": const(0.4)},               # rounds to 0 s
        ("digits", "pacmap"): {"runtime": const(1.6)},                      # rounds to 2 s
        ("digits", "sgd_mmds"): {"runtime": spread(73, 2)},
        ("tasic", "sgd_mmds"): {"runtime": spread(240, 10)},
        ("mnist", "tsne"): {"runtime": spread(14 * 60, 45)},                # std 45 s = 0.75 min -> 1
    }
    build_csv(tmp_path, overrides)
    out, _ = run_tables(monkeypatch, capsys, tmp_path, "runtime")
    rows = {" ".join(line.split()[:2]) if "(" in line else line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "0 s" in rows["Classical"] and "0 ±" not in rows["Classical"]                     # deterministic, identical: no std
    assert "2 ± 0 s" in rows["PaCMAP"] and "73 ± 2 s" in rows["Metric (SGD)"]               # stochastic: std even if all seeds agree
    assert "4 ± 0 m" in rows["Metric (SGD)"] and "14 ± 1 m" in rows["t-SNE"]   # whole minutes on tasic and mnist
    assert "--" in rows["Metric (SMACOF)"]                                       # SMACOF is not run on tasic/mnist

    out, _ = run_tables(monkeypatch, capsys, tmp_path, "runtime", "--latex")
    lines = out.splitlines()
    assert lines[:3] == [r"\toprule", r"& Blobs & Gen. & Digits & Tasic & MNIST \\", r"\midrule"]
    zero = r"10 {\scriptsize$\pm$0} s"
    assert rf"Metric (SGD) & {zero} & {zero} & 73 {{\scriptsize$\pm$2}} s & 4 {{\scriptsize$\pm$0}} m & 0 {{\scriptsize$\pm$0}} m \\" in lines
    tsne = rf"$t$-SNE & {zero} & {zero} & {zero} & 0 {{\scriptsize$\pm$0}} m & 14 {{\scriptsize$\pm$1}} m \\"
    assert tsne in lines
    assert lines[lines.index(tsne) - 1] == r"\midrule"
    for line in lines[3:-1]:
        if line != r"\midrule":
            cells = line.removesuffix(r" \\").split(" & ")[1:]
            for index, cell in enumerate(cells):                                 # blobs, genomes, digits, tasic, mnist
                assert cell == "--" or cell.endswith(" m" if index >= 3 else " s"), (line, cell)


def test_missing_results_are_marked_and_warned_about(tmp_path, monkeypatch, capsys):
    build_csv(tmp_path, {}, drop={("mnist", "umap")})
    out, err = run_tables(monkeypatch, capsys, tmp_path, "triplet_acc", "--latex")
    umap = [l for l in out.splitlines() if l.startswith("UMAP")][0]
    assert umap.removesuffix(r" \\").split(" & ")[-1] == "??"
    assert "warning: 1 cell(s) are missing" in err


def test_summary_has_no_latex_version(tmp_path, monkeypatch, capsys):
    build_csv(tmp_path, {})
    monkeypatch.setattr(tables, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["tables.py", "summary", "--latex"])
    with pytest.raises(SystemExit) as exit_info:
        tables.main()
    assert exit_info.value.code == 2
    assert "not available for the summary" in capsys.readouterr().err


def test_std_in_scriptsize_drops_only_the_leading_zero_of_decimals():
    assert tables._with_std("0.68", "0.01", True) == r"0.68 {\scriptsize$\pm$.01}"
    assert tables._with_std("0.68", "0.00", True) == r"0.68 {\scriptsize$\pm$.00}"
    assert tables._with_std("0.245", "0.012", True) == r"0.245 {\scriptsize$\pm$.012}"
    assert tables._with_std("73", "0", True) == r"73 {\scriptsize$\pm$0}"       # integer runtime stds keep their 0
    assert tables._with_std("73", "10", True) == r"73 {\scriptsize$\pm$10}"
    assert tables._with_std("1.20", "1.25", True) == r"1.20 {\scriptsize$\pm$1.25}"  # a std above 1 is untouched
    assert tables._with_std("0.68", "0.01", False) == "0.68 ± 0.01"


def test_stochastic_methods_always_show_a_std_deterministic_ones_only_if_it_is_above_zero(tmp_path, monkeypatch, capsys):
    perfect = {"knn_acc": const(1.0)}
    one_miss = {"knn_acc": [1.0, 0.998, 1.0]}
    overrides = {("blobs", m): perfect for m in ("classical_mds", "metric_mds", "tsne", "umap")}
    overrides[("blobs", "pacmap")] = one_miss
    build_csv(tmp_path, overrides)
    # umap has a single seed: one value has no spread
    df = pd.read_csv(tmp_path / "all_runs.csv")
    df = df[~((df.dataset == "blobs") & (df.method == "umap") & (df.seed > 0))]
    df.to_csv(tmp_path / "all_runs.csv", index=False)

    plain, _ = run_tables(monkeypatch, capsys, tmp_path, "knn_acc")
    blobs = {line.split()[0]: line.split()[1:4] for line in plain.splitlines()[1:] if line.strip()}   # first cells of each row
    assert blobs["Classical"][0] == "1.00*" and blobs["Metric"][0] == "1.00*"                  # deterministic, identical: bare
    assert blobs["t-SNE"][:3] == ["1.00", "±", "0.00*"]                                        # stochastic, identical: shown
    assert blobs["UMAP"][0] == "1.00*"                                                         # stochastic but one seed: bare
    assert blobs["PaCMAP"][:3] == ["1.00", "±", "0.00*"]                                       # a miss in one seed

    latex, _ = run_tables(monkeypatch, capsys, tmp_path, "knn_acc", "--latex")
    cell = {l.split(" & ")[0]: l.split(" & ")[1] for l in latex.splitlines() if " & " in l and not l.startswith("&")}
    assert cell["$t$-SNE"] == r"\textbf{1.00} {\scriptsize$\pm$.00}"
    assert cell["Classical"] == r"\textbf{1.00}"


def write_isotonic_csv(tmp_path):
    """The numbers of the isotonic-frequency table in the manuscript (times converted to seconds)."""
    blobs = {-100: (76, 0.249), -10: (80, 0.247), 0: (80, 0.249), 10: (83, 0.247), 1: (118, 0.245)}
    mnist = {-10: (11 * 60 + 32, 0.297), 0: (10 * 60 + 22, 0.293), 10: (10 * 60 + 57, 0.293), 1: (17 * 60 + 38, 0.292)}  # -100 is skipped
    rows = []
    for dataset, values in (("blobs", blobs), ("mnist", mnist)):
        for point, (seconds, loss) in values.items():
            for seed, delta in enumerate((-1, 0, 1)):
                rows.append({"dataset": dataset, "method": "sgd_nmds", "seed": seed, "emb_seed": seed, "n": 100,
                             "runtime": seconds + 2 * delta, "isotonic_fit_how_often": point,
                             "stress1_nonmetric": loss + 0.001 * delta})
    (tmp_path / "ablations").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "ablations" / "isotonic_frequency.csv", index=False)


def test_isotonic_table_latex_is_in_seconds_with_small_stds(tmp_path, monkeypatch, capsys):
    write_isotonic_csv(tmp_path)
    out, err = run_tables(monkeypatch, capsys, tmp_path, "isotonic", "--latex")
    assert out.splitlines() == [
        r"\midrule",
        r"Once every 100 epochs & 76 {\scriptsize$\pm$2} s & 0.249 {\scriptsize$\pm$.001} & --- & --- \\",
        r"Once every 10 epochs & 80 {\scriptsize$\pm$2} s & 0.247 {\scriptsize$\pm$.001} & 692 {\scriptsize$\pm$2} s & 0.297 {\scriptsize$\pm$.001} \\",
        r"Once every epoch (default) & 80 {\scriptsize$\pm$2} s & 0.249 {\scriptsize$\pm$.001} & 622 {\scriptsize$\pm$2} s & 0.293 {\scriptsize$\pm$.001} \\",
        r"Once every 10 batches & 83 {\scriptsize$\pm$2} s & 0.247 {\scriptsize$\pm$.001} & 657 {\scriptsize$\pm$2} s & 0.293 {\scriptsize$\pm$.001} \\",
        r"Every batch & 118 {\scriptsize$\pm$2} s & 0.245 {\scriptsize$\pm$.001} & 1058 {\scriptsize$\pm$2} s & 0.292 {\scriptsize$\pm$.001} \\",
        r"\bottomrule",
    ]
    assert err == ""


def test_isotonic_table_plain_has_stds_and_seconds_on_mnist_too(tmp_path, monkeypatch, capsys):
    write_isotonic_csv(tmp_path)
    out, _ = run_tables(monkeypatch, capsys, tmp_path, "isotonic")
    rows = {line[:26].strip(): line for line in out.splitlines()[1:] if line.strip()}
    assert "76 ± 2 s" in rows["Once every 100 epochs"] and "0.249 ± 0.001" in rows["Once every 100 epochs"]
    assert "692 ± 2 s" in rows["Once every 10 epochs"] and " m" not in rows["Once every 10 epochs"]   # seconds, not minutes
    assert "--" in rows["Once every 100 epochs"] and "---" not in rows["Once every 100 epochs"]       # skipped on MNIST
