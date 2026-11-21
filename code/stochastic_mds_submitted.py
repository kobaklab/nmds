"""Stochastic (mini-batch) metric and non-metric MDS, optimized with Adam in PyTorch."""
import time

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.manifold import ClassicalMDS


def stochasticMDS(
    X,
    n_components=2,
    precomputed_distances=False,
    non_metric=False,
    verbose=True,
    n_epochs="auto",
    batch_size="auto",
    print_every_epoch="auto",
    lr=0.02,
    isotonic_fit_how_often=0,
    init_scale=0.5,
    init=None,
    seed=42,
):
    """Embed X with mini-batch SGD on the (non-)metric MDS stress.

    X: data matrix (n, d), or an (n, n) distance matrix if precomputed_distances.
    non_metric: if True, the high-dimensional distances in each batch are first
        passed through an isotonic regression fitted to the embedding distances.
    n_epochs, batch_size: "auto" gives 200 epochs / batch 100 up to n = 5000,
        200 epochs / batch 1000 up to n = 50 000, and 20 epochs / batch 1000 above.
    isotonic_fit_how_often: when the isotonic regression is refitted (non-metric
        only; it is applied on every batch regardless).
          0    on the first batch of every epoch
          k>0  every k batches
          k<0  on the first batch of every |k|-th epoch
    init: initial embedding (n, n_components); by default PCA (or classical MDS for
        precomputed distances), scaled to unit std of the first coordinate times init_scale.
    seed: seeds the batch shuffling and the PCA initialization (the only random parts).

    In both modes the (transformed) distances of each batch are divided by a coefficient
    such that their mean squared off-diagonal value is 1 (as in sklearn's non-metric MDS).
    It is computed on the first batch of each epoch and used for the whole epoch. This makes
    the optimization independent of the scale of the data, so lr and init_scale need no tuning;
    the returned embedding is on this normalized scale, not on the scale of the data
    (unlike the output of sklearn's metric MDS).

    The loss that is minimized is the raw stress. Returns the embedding and, for
    monitoring, the stress-1 averaged over the batches of each epoch. The latter is
    computed on the fly on normalized mini-batch distances, so use a separate
    evaluation for the final stress-1.
    """

    sample_size = X.shape[0]

    if batch_size == "auto":
        if sample_size <= 5000:
            batch_size = 100
        else:
            batch_size = 1000
            
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

    generator = torch.Generator().manual_seed(seed)

    model = torch.nn.Embedding.from_pretrained(
        torch.tensor(Z.astype(np.float32)), freeze=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if non_metric:
        ir = IsotonicRegression(out_of_bounds="clip")

    epochs_stress1 = np.zeros(n_epochs)
    n_batches = sample_size // batch_size
    
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
                dists = cdist(X[ind_np], X[ind_np])

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

            # Loss function = raw stress
            raw_stress = torch.mean((model_dists - dists)**2)

            raw_stress.backward()
            optimizer.step()

            # Only for monitoring, not used for optimization:
            # this is sqrt of "normalized stress" (Borg & Groenen)
            # After convergence it is equal to the optimal stress 1
            stress1 = np.sqrt(raw_stress.item() / np.mean(dists.detach().numpy() ** 2))
            
            epoch_stress1 += stress1.item()

        scheduler.step()
        epoch_stress1 /= n_batches
        t_end = time.time()

        epochs_stress1[epoch] = epoch_stress1
        if verbose and (epoch + 1) % print_every_epoch == 0:
            print(f"Epoch {epoch + 1:3}, stress-1 {epoch_stress1:.3f}, time {t_end - t_start:.1f} s")

    Z = model.weight.detach().numpy()
    return Z, epochs_stress1
