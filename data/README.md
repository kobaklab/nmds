Input data (not tracked by git). Expected files:

    genomes-data/gt_sum_thinned.npz
    genomes-data/affy_samples.20141118.panel
    genomes-data/population_colors.csv
    genomes-data/20131219.populations.tsv     population descriptions; not used by the code
    tasic-data/tasic-pca50.npy
    tasic-data/tasic-colors.npy
    tasic-data/tasic-ttypes.npy               only used by the Tasic Shepard figure

MNIST is downloaded automatically (into `openml/`) on first use; blobs and digits are generated.

Copy all files from these two folders:

* https://github.com/kobaklab/low-dim-embeddings-review/tree/main/data/1000-Genomes-Project into `genomes-data/`
* https://github.com/berenslab/rna-seq-tsne/tree/master/data/tasic-preprocessed into `tasic-data/`
