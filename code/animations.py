"""Animations of the optimization, for talks: SGD (ours) and SMACOF (sklearn), metric and non-metric.

  uv run python code/animations.py compute                  # results/animations/<dataset>_<method>.npz
  uv run python code/animations.py render                   # figures/animations/<dataset>_<run>_dark.gif
  uv run python code/animations.py render --light           # ... _light.gif
  uv run python code/animations.py render --jumps           # no interpolation: ... _dark_jumps.gif

Runs a dataset skips (SMACOF on Tasic and
MNIST) are left out.

Runs: sgd_mmds, sgd_nmds (our SGD), smacof_mmds, smacof_nmds (sklearn).

Both stages take --datasets (default: genomes) and --methods (default: all four below). Like the rest of the
pipeline, `compute` skips trajectories that are already saved (--force recomputes); `render` reads them.

compute
  SGD     stochasticMDS with the same settings as the main runs (methods.py; seed 0), with a snapshot of
          the embedding every k mini-batches, with k as small as allowed by at most MAX_SNAPSHOTS = 2000
          snapshots and MAX_POINTS_IN_SNAPSHOTS = 25 million points in all of them together
          (genomes: k = 1 for metric, 4 for non-metric; Tasic: 5 and 11).
  SMACOF  sklearn's MDS with the same settings as the main runs, every iteration recorded by
          smacof_trajectory.record_smacof (see there why not one-iteration-at-a-time calls).
  For every whole epoch (SGD; the first snapshot at or after it; ~200 log-spaced epochs for longer runs) or
  every iteration (SMACOF) the evaluated stress-1 is saved too, computed as in
  metrics.py (metric stress-1 for the metric runs, Kruskal's for the non-metric ones) on up to 5000 points.

render
  One GIF per run: the embedding, no title, the epoch or iteration counter top right ("12.5 / 200") and the
  current stress-1 bottom right. What is shown:
  * Every frame is rescaled to unit RMS radius and centred on the middle of its (robust) bounding box. SMACOF's metric runs are on the scale of the
    data and its non-metric ones on the normalized scale (and its first frame, the classical-MDS start, on
    the scale of the data again); without this the picture would jump. Only the shape is shown.
  * Time is logarithmic: the animation shows log(1 + epoch / tau) (or iteration), tau = 2 by default, so the
    first few epochs, where most of the change happens, take about a quarter of the video, and the last
    half of the run (the annealing of the learning rate, which sharpens the SGD picture) a few seconds.
    --tau 0 gives linear time. (Pacing by the amount of motion does not work for SGD: its epoch-to-epoch
    jitter keeps the motion high long after the shape has settled.) Between saved snapshots positions are
    interpolated linearly (--jumps: every snapshot is shown as it is until the next one, so every SMACOF
    iteration is a jump). The counter shows the true
    epoch or iteration.
  * Stress-1 is the evaluated value saved by compute, linearly interpolated between whole epochs.
"""
import argparse

import numpy as np

import paths
from datasets import DATASETS

ANIMATIONS = paths.RESULTS / "animations"
VIDEOS = paths.FIGURES / "animations"

# run name -> optimizer, kind of MDS, and the method in methods.py whose settings it uses
RUNS = {
    "sgd_mmds": dict(optimizer="sgd", non_metric=False, method="sgd_mmds"),
    "sgd_nmds": dict(optimizer="sgd", non_metric=True, method="sgd_nmds"),
    "smacof_mmds": dict(optimizer="smacof", non_metric=False, method="metric_mds"),
    "smacof_nmds": dict(optimizer="smacof", non_metric=True, method="nonmetric_mds"),
}
SEED = 0
MAX_SNAPSHOTS = 2000  # SGD: a snapshot every k batches, with k as small as this allows (k = 1 on genomes metric)
MAX_POINTS_IN_SNAPSHOTS = 25_000_000  # and at most this many points in all snapshots together (memory: Tasic)


def path_for(dataset, run):
    return ANIMATIONS / f"{dataset}_{run}.npz"


# ---------------------------------------------------------------------------------------------- compute


def _n_batches(n, non_metric, kwargs):
    """Total number of mini-batches of a stochasticMDS run with these settings (its "auto" rules)."""
    batch_size = kwargs.get("batch_size", "auto")
    if batch_size == "auto":
        batch_size = min(100 if n <= 5000 and non_metric else 1000, n)
    n_epochs = kwargs.get("n_epochs", "auto")
    if n_epochs == "auto":
        n_epochs = 200 if n <= 50_000 else 20
    return n_epochs * (n // batch_size)


def _stress_curve(ctx, frames, non_metric):
    from metrics import stress1_metric, stress1_nonmetric

    fn = stress1_nonmetric if non_metric else stress1_metric
    return np.array([fn(ctx, np.asarray(Z, dtype=float)) for Z in frames])


def compute(dataset, method, X, y, precomputed):
    from metrics import EvalContext

    run = RUNS[method]
    if run["optimizer"] == "sgd":
        from stochastic_mds import stochasticMDS

        kwargs = DATASETS[dataset].get("method_kwargs", {}).get(run["method"], {})
        n_batches = _n_batches(len(X), run["non_metric"], kwargs)
        max_snapshots = min(MAX_SNAPSHOTS, MAX_POINTS_IN_SNAPSHOTS // len(X))
        every = int(np.ceil(n_batches / max_snapshots))
        _, _, snaps = stochasticMDS(
            X, precomputed_distances=precomputed, non_metric=run["non_metric"], seed=SEED, verbose=False,
            snapshot_every=every, snapshot_unit="batch", **kwargs,
        )
        frames, progress, unit = snaps["Z"], snaps["epoch"], "epoch"
        # stress-1 at the first snapshot at or after every whole epoch (exactly the whole epochs if every = 1);
        # for long runs at ~200 epochs, spaced like the animation's logarithmic time
        epochs = np.arange(np.ceil(progress[-1]) + 1)
        if len(epochs) > 201:
            epochs = np.unique(np.concatenate([[0], np.round(np.geomspace(1, epochs[-1], 200))]))
        at_curve = np.unique(np.searchsorted(progress, epochs - 1e-9))
    else:
        from smacof_trajectory import record_smacof

        if not precomputed:
            from sklearn.metrics import pairwise_distances
            X = pairwise_distances(X)
        frames, _, _ = record_smacof(X, metric_mds=not run["non_metric"])
        frames = frames.astype(np.float32)
        progress, unit = np.arange(len(frames), dtype=float), "iteration"
        at_curve = np.arange(len(frames))

    ctx = EvalContext(X, y, precomputed or run["optimizer"] == "smacof", SEED)
    stress = _stress_curve(ctx, frames[at_curve], run["non_metric"])
    out = path_for(dataset, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frames=frames, progress=progress, unit=unit,
             stress_progress=progress[at_curve], stress=stress,
             y=np.asarray(y).astype(str) if np.asarray(y).dtype == object else y)  # colors: no pickled arrays
    print(f"{dataset} {method}: {len(frames)} frames, stress-1 {stress[0]:.3f} -> {stress[-1]:.3f}  ->  {out}")


# ---------------------------------------------------------------------------------------------- render


class Trajectory:
    """A saved run, normalized for display and resampled to video frames."""

    def __init__(self, dataset, method, tau=2.0, jumps=False):
        with np.load(path_for(dataset, method)) as f:
            self.frames = f["frames"]  # float32 (Tasic: ~1000 frames of 24k points)
            self.progress, self.unit = f["progress"], str(f["unit"])
            self.stress_progress, self.stress, self.y = f["stress_progress"], f["stress"], f["y"]
        self.method, self.jumps = method, jumps

        # Logarithmic time: video time t in [0, 1] is log(1 + progress / tau), normalized.
        self.tau = tau
        self.clock = self.warp(self.progress)

        # View of every frame: scaled to unit RMS radius and centred on the middle of its robust bounding box
        # (0.5% and 99.5% quantiles; the centroid would be pulled away from the middle of the picture, e.g. by
        # the dense neuronal cells in Tasic). In place, to keep memory low.
        for F in self.frames:
            F -= F.mean(axis=0)
            F /= np.sqrt((F ** 2).sum(axis=1).mean())
            F -= np.quantile(F, [0.005, 0.995], axis=0).mean(axis=0)

    def warp(self, progress):
        """Video time in [0, 1] of an epoch or iteration."""
        last = self.progress[-1]
        if self.tau > 0:
            return np.log1p(np.asarray(progress) / self.tau) / np.log1p(last / self.tau)
        return np.asarray(progress) / last

    def at(self, t):
        """Positions and progress at normalized time t in [0, 1] (linear interpolation)."""
        i = int(np.clip(np.searchsorted(self.clock, t, side="right") - 1, 0, len(self.clock) - 2))
        w = 0.0 if self.clock[i + 1] == self.clock[i] else (t - self.clock[i]) / (self.clock[i + 1] - self.clock[i])
        w = float(np.clip(w, 0, 1))
        if self.jumps:  # show every snapshot as it is, until the next one
            i = i + 1 if w >= 1 else i
            return self.frames[i], self.progress[i]
        Z = (1 - w) * self.frames[i] + w * self.frames[i + 1]
        progress = (1 - w) * self.progress[i] + w * self.progress[i + 1]
        return Z, progress

    def counter(self, progress):
        if self.unit == "epoch":
            return f"{progress:.1f} / {self.progress[-1]:.0f}"
        return f"{int(np.floor(progress + 1e-9))} / {self.progress[-1]:.0f}"

    def limit(self, quantile=0.995):
        """Half-width of the axes: large enough for nearly all points in every frame."""
        return max(float(np.quantile(np.abs(F).max(axis=1), quantile)) for F in self.frames) * 1.05


THEMES = {
    "light": dict(bg="white", fg="0.3"),
    "dark": dict(bg="#111111", fg="0.75"),
}


def render(tr, out, seconds=12.0, fps=15, hold_start=1.0, hold_end=2.5, size=720, point_size=None,
           theme="light"):
    """Write the trajectory tr as a square GIF of size x size pixels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    th = THEMES[theme]
    dpi = 100
    scale = size / 720
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=th["bg"])
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_facecolor(th["bg"])
    lim = tr.limit()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()

    Z0, p0 = tr.at(0)
    cmap = "tab10" if np.issubdtype(tr.y.dtype, np.number) else None  # class labels, not colors
    s = point_size if point_size is not None else 4 * scale ** 2 * min(1, np.sqrt(4000 / len(tr.y)))
    sc = ax.scatter(Z0[:, 0], Z0[:, 1], s=s, c=tr.y, cmap=cmap, ec="none")
    text = dict(color=th["fg"], fontsize=15 * scale, family="monospace")
    counter = fig.text(0.97, 0.97, tr.counter(p0), ha="right", va="top", **text)
    stress = fig.text(0.97, 0.03, "", ha="right", va="bottom", **text)

    n_move = int(round(seconds * fps))
    times = np.concatenate([np.zeros(int(hold_start * fps)), np.linspace(0, 1, n_move),
                            np.ones(int(hold_end * fps))])

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, out, dpi):
        for t in times:
            Z, p = tr.at(t)
            sc.set_offsets(Z)
            counter.set_text(tr.counter(p))
            stress.set_text(f"Stress-1: {np.interp(p, tr.stress_progress, tr.stress):.2f}")
            writer.grab_frame(facecolor=th["bg"])
    plt.close(fig)
    print(f"  {out}  ({len(times) / fps:.1f} s, {out.stat().st_size / 1e6:.1f} MB)")


# Cells whose colours --lighten mixes with white, per dataset: the non-neuronal cells of Tasic have dark grey-brown
# colours that disappear on a dark background (e.g. on a projector).
TASIC_NON_NEURONAL = {"Astro", "Endo", "Microglia", "OPC", "Oligo", "PVM", "Peri", "SMC", "VLMC"}


def _lighten_mask(dataset):
    if dataset == "tasic":
        ttypes = np.load(paths.DATA / "tasic-data/tasic-ttypes.npy", allow_pickle=True)
        return np.array([str(t).split()[0] in TASIC_NON_NEURONAL for t in ttypes])
    raise ValueError(f"--lighten is only defined for tasic, not for {dataset}")


def lighten(colors, mask, amount=0.5):
    """Colours of the cells in mask mixed with white (amount 0: unchanged, 1: white)."""
    import matplotlib.colors as mcolors

    out = np.array([mcolors.to_hex(mcolors.to_rgb(c)) for c in colors], dtype=object)
    for c in np.unique(out[mask]):
        rgb = np.array(mcolors.to_rgb(c))
        out[mask & (out == c)] = mcolors.to_hex(rgb + amount * (1 - rgb))
    return out.astype(str)


def _runs_on(dataset, method):
    """False for runs the dataset skips (SMACOF on Tasic and MNIST: too slow)."""
    return RUNS[method]["method"] not in DATASETS[dataset].get("skip", [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["compute", "render"])
    parser.add_argument("--datasets", nargs="+", default=["genomes"])
    parser.add_argument("--methods", nargs="+", default=list(RUNS), choices=list(RUNS))
    parser.add_argument("--force", action="store_true", help="compute: recompute saved trajectories")
    parser.add_argument("--seconds", type=float, default=12.0, help="render: duration of the motion")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--size", type=int, default=720, help="render: width = height of the GIF in pixels")
    parser.add_argument("--tau", type=float, default=2.0,
                        help="render: video time ~ log(1 + epoch / tau); 0 = linear time")
    parser.add_argument("--light", action="store_true", help="render: white background instead of dark")
    parser.add_argument("--lighten", action="store_true",
                        help="render: lighter colours for the non-neuronal cells of Tasic (..._lighter.gif)")
    parser.add_argument("--jumps", action="store_true",
                        help="render: no interpolation between snapshots (SMACOF: one jump per iteration)")
    args = parser.parse_args()

    for dataset in args.datasets:
        if args.stage == "compute":
            todo = [m for m in args.methods if args.force or not path_for(dataset, m).exists()]
            if not todo:
                print(f"{dataset}: all trajectories saved (--force recomputes)")
                continue
            todo = [m for m in todo if _runs_on(dataset, m)]
            if not todo:
                continue
            X, y, precomputed = DATASETS[dataset]["load"]()
            for method in todo:
                compute(dataset, method, X, y, precomputed)
            continue

        theme = "light" if args.light else "dark"
        for method in [m for m in args.methods if _runs_on(dataset, m)]:
            if not path_for(dataset, method).exists():
                print(f"  {dataset} {method}: no saved trajectory, run `compute` first")
                continue
            suffix = f"_{theme}" + ("_jumps" if args.jumps else "") + ("_lighter" if args.lighten else "")
            tr = Trajectory(dataset, method, args.tau, args.jumps)
            if args.lighten:
                tr.y = lighten(tr.y, _lighten_mask(dataset))
            render(tr, VIDEOS / f"{dataset}_{method}{suffix}.gif",
                   seconds=args.seconds, fps=args.fps, size=args.size, theme=theme)


if __name__ == "__main__":
    main()
