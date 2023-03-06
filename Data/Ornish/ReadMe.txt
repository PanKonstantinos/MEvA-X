In this folder there are 3 different files for the data used in the present work to produce the results of the MEvA-X tool.

In this directory you can find the following:
- diet_dataset.txt contains the data normalized, with a preprocessing step of removing the batch effect from the different experiments. In addition, it containss the Age [float], Gender(Sex) [binary], COPD [binary], and Diabetes [binary] of the samples as they are in the original dataset.

In the subfolder "alternarive_data" there are the following files:
- Original_data.txt are the data normalized from NCBI with the addition of Age [float], Gender(Sex) [binary], COPD [binary], and Diabetes [binary].
- diet_dataset_batch_effect_removed.txt contains the data normalized from NCBI but with a preprocessing step of removing the batch effect from the different experiments. In addition, it containss the Age, Gender(Sex), COPD, and Diabetes of the samples normalized but also affected by the batch effect removing step.

In the subfolder FS_methods there are thi files with the precalculated features selected by mRMR, SKB, Wilcoxon sum rank and JMI algorithms.

In the GSE66175_RAW subfolder, there are the raw data from GEO reposiory with the expression values for the samples.