Input data (not tracked by git). Expected files:

    genomes-data/1000-genomes-D.npy        distance matrix, condensed (n(n-1)/2 values, scipy pdist format)
                                           or full n x n, any float precision (n = 3450: both are ~47.6 MB
                                           as float64-condensed or float32-full)
    genomes-data/1000-genomes-labels.npy
    tasic-data/tasic-pca50.npy
    tasic-data/tasic-colors.npy
    tasic-data/tasic-ttypes.npy            only used by the Tasic Shepard figure

MNIST is downloaded automatically (into `openml/`) on first use; blobs and digits are generated.

Preprocessed Tasic data can be downloaded from https://github.com/berenslab/rna-seq-tsne/tree/master/data/tasic-preprocessed.

Preprocessed Genomes data will be shared later on.
