"""How datasets and methods are presented in tables and figures: names, order, and
which method fills which row. Shared by tables.py and the figure notebooks."""

DATASET_LABELS = {
    "blobs": "Blobs",
    "genomes": "Gen.",
    "digits": "Digits",
    "tasic": "Tasic",
    "mnist": "MNIST",
}

# A table row is a label, not a method: "Metric" and "Non-metric" are the sklearn
# (SMACOF) implementations on the small datasets and our SGD on the large ones.
ROWS = {
    "Classical": dict(default="classical_mds"),
    "Metric": dict(default="metric_mds", tasic="sgd_mmds", mnist="sgd_mmds"),
    "Non-metric": dict(default="nonmetric_mds", tasic="sgd_nmds", mnist="sgd_nmds"),
    "t-SNE": dict(default="tsne"),
    "UMAP": dict(default="umap"),
    "PaCMAP": dict(default="pacmap"),
}

# One row per method (used by the runtime table), in this order.
METHOD_LABELS = {
    "classical_mds": "Classical",
    "metric_mds": "Metric (SMACOF)",
    "sgd_mmds": "Metric (SGD)",
    "nonmetric_mds": "Non-metric (SMACOF)",
    "sgd_nmds": "Non-metric (SGD)",
    "tsne": "t-SNE",
    "umap": "UMAP",
    "pacmap": "PaCMAP",
}


def method_for(row, dataset):
    spec = ROWS[row]
    return spec.get(dataset, spec["default"])
