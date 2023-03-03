# MEvA-X
An open source biomarker discovery tool based on a multi-objective Evolutionary algorithm and the XGBoost Classifier.
Benchmarked on 2 datasets. One omics and one clinical. With MEvA-X the performance of the XGBoost Classifiers improved their overall performance and/or the simplicity of the final models. 


 <table>
  <tr>
    <th>Python version</th>
    <th>3.9</th>
    <th>https://www.python.org/</th>
  </tr>
  </table>

  <table>
Dependencies:
  <tr>
    <th>Library</th>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Pandas</td>
    <td>>1.3.0</td>
    <td>https://pypi.org/project/pandas/</td>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>1.7.3</td>
    <td>https://pypi.org/project/xgboost/</td>
  </tr>
  <tr>
    <td>Numpy</td>
    <td>1.21.5</td>
    <td>https://numpy.org/</td>
  </tr>
  <tr>
    <td>Sklearn</td>
    <td>0.24.2</td>
    <td>https://scikit-learn.org/stable/</td>
  </tr>
  <tr>
    <td>mifs</td>
    <td>0.0.1.dev0</td>
    <td>https://github.com/danielhomola/mifs</td>
  </tr>
  <tr>
    <td>pickle5</td>
    <td></td>
    <td>https://pypi.org/project/pickle5/</td>
  </tr>
  <tr>
    <td>knnimpute</td>
    <td>0.1.0</td>
    <td>https://github.com/iskandr/knnimpute</td>
  </tr>
  <tr>
    <td>requests</td>
    <td>2.28.1</td>
    <td>https://pypi.org/project/requests/</td>
  </tr>
</table>

<h2>Example of calling MEvA-X from terminal:</h2>

```
python MEvA-X_V1.0.0.py
```

<h3><i>(Comming soon)</i> Example of calling MEvA-X_V2 from terminal:</h3>

```
python MEvA-X.py -K 10 -P 50 -G 200 --dataset my_data.txt --labels my_labels.tsv -FS precalculated_features.csv --output_dir current_folder -cop 0.9 -acp 0 -mp 0.1 -goal_sig_lst 0.8 2 0.8 1 1 0.7 0.7 1 2 0.5 2
```


Parameters of the algorithm:
<table>
  <tr>
    <th>Parameter name</th>
    <th>short description</th>
    <th>Default value</th>
  </tr>
  <tr>
    <td>Population</td>
    <td>Number of individuals for the evolutionary algorithm</td>
    <td>50</td>
  </tr>
  <tr>
    <td>Generations</td>
    <td>Number of gererations the evolutionary will run for</td>
    <td>100</td>
  </tr>
  <tr>
    <td>dataset_filename</td>
    <td>Name of the data in the current directory</td>
    <td><i>None</i></td>
  </tr>
  <tr>
    <td>labels_filename</td>
    <td>Name of the file that contains the labels of the datapoints</td>
    <td><i>None</i></td>
  </tr>
  <tr>
    <td>two_points_crossover_probability</td>
    <td>probability of the offsprings to be produced by the exchange of pieces from the parental individuals</td>
    <td>0.1 (10%)</td>
  </tr>
  <tr>
    <td>arithmetic_crossover_probability</td>
    <td>probability of an arithmetic crossover of the parental individuals to produce the offsprings</td>
    <td>0.0 (0%)</td>
  </tr>
  <tr>
    <td>mutation_probability</td>
    <td>probability of an offspring to mutate</td>
    <td>0.05 (5%)</td>
  </tr>
  <tr>
    <td>goal_significances_filename</td>
    <td>array of weights for the objectives</td>
    <td>array of ones in the length of the objectives</td>
  </tr>
 <tr>
    <td>num_of_folds</td>
    <td>Number of folds for cross validation</td>
    <td>10</td>
  </tr>
</table>
