# MEvA-X
An open source biomarker discovery tool based on a multi-objective Evolutionary algorithm and the XGBoost Classifier.
Benchmarked on 2 datasets. One omics and one clinical. With MEvA-X the performance of the XGBoost Classifiers improved their overall performance and/or the simplicity of the final models. 

Dependencies:

- Pandas
- XGBoost
- mifs (https://github.com/danielhomola/mifs)
- pickle
- knnimpute (https://github.com/iskandr/knnimpute)
- requests

Example for calling MEvA-X_V2 from terminal:
python MEvA-X.py

(Comming soon) Example for calling MEvA-X_V2 from terminal:
```
python MEvA-X.py -K 10 -P 50 -G 200 --dataset my_data.txt --labels my_labels.tsv -FS precalculated_features.csv --output_dir current_folder -cop 0.9 -acp 0 -mp 0.1 -goal_sig_lst 0.8 2 0.8 1 1 0.7 0.7 1 2 0.5 2
```
