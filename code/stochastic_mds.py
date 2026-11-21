"""Stochastic (mini-batch) metric and non-metric MDS: SGD in PyTorch with the step size of SMACOF."""
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import ClassicalMDS
from sklearn.metrics import euclidean_distances


def stochasticMDS(
    X,
    n_components=2,
    precomputed_distances=False,
    non_metric=False,
    verbose=True,
    n_epochs="auto",
    batch_size="auto",
    print_every_epoch="auto",
    isotonic_fit_how_often=0,
    init_scale=0.5,
    init=None,
    seed=42,
    snapshot_every=None,
    snapshot_unit="epoch",
    compute_all_distances="auto",
):
    """Embed X with mini-batch SGD on the (non-)metric MDS stress.

    X: data matrix (n, d), or an (n, n) distance matrix if precomputed_distances.
    non_metric: if True, the high-dimensional distances in each batch are first
        passed through an isotonic regression fitted to the embedding distances.
    n_epochs, batch_size: "auto" gives 200 epochs up to n = 50 000 and 20 epochs above;
        batch 100 for non-metric MDS up to n = 5000, and batch 1000 otherwise (at most n).
    isotonic_fit_how_often: when the isotonic regression is refitted (non-metric
        only; it is applied on every batch regardless).
          0    on the first batch of every epoch
          k>0  every k batches
          k<0  on the first batch of every |k|-th epoch
    init: initial embedding (n, n_components); by default PCA (or classical MDS for
        precomputed distances), scaled to unit std of the first coordinate times init_scale.
    seed: seeds the batch shuffling and the PCA initialization (the only random parts).
    snapshot_every: None (default) or an int N: also return a copy of the embedding every
        N epochs (snapshot_unit="epoch") or every N mini-batches (snapshot_unit="batch"),
        e.g. for animations. The initialization and the final embedding are always included.
    compute_all_distances: if True, the full (n, n) distance matrix of a data matrix X is
        computed once instead of the distances of every batch; "auto" does this up to n = 10 000.

    In both modes the (transformed) distances of each batch are divided by a coefficient
    such that their mean squared off-diagonal value is 1 (as in sklearn's non-metric MDS).
    It is computed on the first batch of each epoch and used for the whole epoch. This makes
    the optimization independent of the scale of the data, so init_scale needs no tuning;
    the returned embedding is on this normalized scale, not on the scale of the data
    (unlike the output of sklearn's metric MDS).

    Each batch takes one plain SGD step on its raw stress (the sum over pairs i < j) with
    step size 1 / (2 * batch_size): one SMACOF step on the batch (the Guttman transform,
    which for unweighted stress equals a gradient step of size 1 / (2n)). The step size is
    annealed to 0 over the epochs with a cosine schedule. Only the points of the batch move.

    Returns the embedding and, for
    monitoring, the stress-1 averaged over the batches of each epoch. The latter is
    computed on the fly on normalized mini-batch distances, so use a separate
    evaluation for the final stress-1.

    With snapshot_every, a third value is returned: a dict with
      "Z"      (n_snapshots, n, n_components) float32, the embeddings
      "epoch"  (n_snapshots,) training progress in epochs: 0 is the initialization, 1 the end
               of the first epoch, 1.5 halfway through the second one
      "lr"     (n_snapshots,) the step size that was in effect for the last step, i.e.
               1 / (2 * batch_size) times the cosine factor (for the initialization:
               1 / (2 * batch_size))
    Taking snapshots does not change the optimization.
    """
    if snapshot_every is not None:
        if snapshot_unit not in ("epoch", "batch"):
            raise ValueError(f'snapshot_unit must be "epoch" or "batch", not {snapshot_unit!r}')
        if int(snapshot_every) != snapshot_every or snapshot_every < 1:
            raise ValueError(f"snapshot_every must be a positive integer or None, not {snapshot_every!r}")

    sample_size = X.shape[0]

    if batch_size == "auto":
        if sample_size <= 5000 and non_metric:
            batch_size = 100
        else:
            batch_size = 1000
        batch_size = min(batch_size, sample_size)
            
    if n_epochs == "auto":
        if sample_size <= 50_000:
            n_epochs = 200
        else:
            n_epochs = 20

    if print_every_epoch == "auto":
        if n_epochs > 20:
            print_every_epoch = 10
        else:
            print_every_epoch = 1

    if init is not None:        
        Z = init
    else:
        if precomputed_distances:
            Z = ClassicalMDS(n_components=n_components, metric="precomputed").fit_transform(X)
        else:
            Z = PCA(n_components=n_components, random_state=seed).fit_transform(X)
        Z /= np.std(Z[:, 0])
        Z *= init_scale

    if compute_all_distances == "auto":
        compute_all_distances = sample_size <= 10_000
    if compute_all_distances and not precomputed_distances:
        X = euclidean_distances(X)
        precomputed_distances = True

    generator = torch.Generator().manual_seed(seed)

    # C-contiguous: classical MDS (the initialization for distance matrices) returns a column-major
    # array, and SGD's sparse update of a non-contiguous weight fails when torch uses several threads
    model = torch.nn.Embedding.from_pretrained(
        torch.tensor(np.ascontiguousarray(Z, dtype=np.float32)), freeze=False, sparse=True
    )

    # one SMACOF (Guttman transform) step on the batch, see the docstring
    optimizer = torch.optim.SGD(model.parameters(), lr=1 / (2 * batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if non_metric:
        ir = IsotonicRegression(out_of_bounds="clip")

    epochs_stress1 = np.zeros(n_epochs)
    n_batches = sample_size // batch_size

    snapshots = {"Z": [], "epoch": [], "lr": []}

    def take_snapshot(progress, lr_in_effect):
        snapshots["Z"].append(model.weight.detach().numpy().copy())
        snapshots["epoch"].append(progress)
        snapshots["lr"].append(lr_in_effect)

    if snapshot_every is not None:
        take_snapshot(0.0, optimizer.param_groups[0]["lr"])

    for epoch in range(n_epochs):
        shuffle = torch.randperm(sample_size, generator=generator)

        epoch_stress1 = 0
        t_start = time.time()

        for batch in range(n_batches):
            ind = shuffle[batch * batch_size : (batch + 1) * batch_size]
            ind_np = ind.numpy()

            if precomputed_distances:
                dists = X[np.ix_(ind_np, ind_np)]
            else:
                # torch rather than sklearn, so that all threads come from torch's pool (sklearn's
                # BLAS threads competing with torch's made this several times slower on many cores)
                batch_X = torch.from_numpy(X[ind_np])
                dists = torch.cdist(batch_X, batch_X).fill_diagonal_(0).numpy()

            optimizer.zero_grad()
            embedding = model(ind)
            model_dists = torch.cdist(embedding, embedding)

            if non_metric:
                if (
                    (isotonic_fit_how_often == 0 and batch == 0) or
                    (isotonic_fit_how_often > 0 and batch % isotonic_fit_how_often == 0) or
                    (isotonic_fit_how_often < 0 and batch == 0 and epoch % -isotonic_fit_how_often == 0)
                ):
                    ir.fit(dists.ravel(), model_dists.detach().numpy().ravel())
                dists = ir.transform(dists.ravel()).reshape(dists.shape)

            if batch == 0:
                # same normalization as in sklearn
                dists_scaling = np.sqrt(np.sum(dists ** 2) / (batch_size ** 2 - batch_size))
            dists = dists / dists_scaling

            dists = torch.tensor(dists.astype(np.float32))

            # Loss function = raw stress, summed over pairs i < j (the matrices are symmetric
            # with zero diagonal, so this is half the sum over the full matrix)
            raw_stress = 0.5 * torch.sum((model_dists - dists)**2)

            raw_stress.backward()
            lr_in_effect = optimizer.param_groups[0]["lr"]
            optimizer.step()

            if snapshot_every is not None:
                n_steps = epoch * n_batches + batch + 1
                if snapshot_unit == "batch" and n_steps % snapshot_every == 0:
                    take_snapshot(n_steps / n_batches, lr_in_effect)
                elif snapshot_unit == "epoch" and batch == n_batches - 1 and (epoch + 1) % snapshot_every == 0:
                    take_snapshot(float(epoch + 1), lr_in_effect)

            # Only for monitoring, not used for optimization:
            # this is sqrt of "normalized stress" (Borg & Groenen)
            # After convergence it is equal to the optimal stress 1
            stress1 = np.sqrt(raw_stress.item() / (0.5 * np.sum(dists.detach().numpy() ** 2)))
            
            epoch_stress1 += stress1.item()

        scheduler.step()
        epoch_stress1 /= n_batches
        t_end = time.time()

        epochs_stress1[epoch] = epoch_stress1
        if verbose and (epoch + 1) % print_every_epoch == 0:
            print(f"Epoch {epoch + 1:3}, stress-1 {epoch_stress1:.3f}, time {t_end - t_start:.1f} s")

    Z = model.weight.detach().numpy()
    if snapshot_every is None:
        return Z, epochs_stress1

    if snapshots["epoch"][-1] != n_epochs:  # always end with the final embedding
        take_snapshot(float(n_epochs), lr_in_effect)
    snapshots = {
        "Z": np.stack(snapshots["Z"]).astype(np.float32),
        "epoch": np.array(snapshots["epoch"]),
        "lr": np.array(snapshots["lr"]),
    }
    return Z, epochs_stress1, snapshots
