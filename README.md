# MEvA-X
An open source biomarker discovery tool based on a multi-objective Evolutionary algorithm and the XGBoost Classifier.
Benchmarked on 2 datasets. One omics and one clinical. With MEvA-X the performance of the XGBoost Classifiers improved their overall performance and/or the simplicity of the final models. 

</h2>Details</h2>
<h4>About the algorithm:</h4>

 <table>
  <tr>
    <th>Python version</th>
    <th>3.9</th>
    <th>https://www.python.org/</th>
  </tr>
  </table>

<h4>Dependencies:</h4>
  <table>
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
    <td>pickle</td>
    <td></td>
    <td>https://docs.python.org/3/library/pickle.html</td>
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

<h1> Tutorial </h1>
<h2>Example of calling MEvA-X from terminal:</h2>

```
python MEvA-X_V1.0.0.py
```

<p>The parameters of the algorithm can be changed directly through the script in the __main__ section for the version V1.0.0.
 See in the table below the details for the parameters the user can experiment with:</p>

<h3>Parameters of the algorithm:</h3>
<table>
  <tr>
   <th>Parameter name</th>
   <th>Short description</th>
   <th>Default value</th>
   <th>Comment</th>
  </tr>
  <tr>
   <td>Population</td>
   <td>Number of individuals for the evolutionary algorithm</td>
   <td>50</td>
   <td>The higher the value the better the space can be explored but the slower the algorithm will get. <i>Advised values: 50-200</i> </td>
  </tr>
  <tr>
   <td>Generations</td>
   <td>Number of gererations the evolutionary will run for</td>
   <td>100</td>
   <td>The higher the value the better the further we allow the algorithm to explore, but the slower the algorithm will get overall. <i>Advised values: 50-200</i> </td>
  </tr>
  <tr>
   <td>dataset_filename</td>
   <td>Name of the data in the current directory</td>
   <td><i>None</i></td>
   <td>The dataset must be in the form of FeaturesXSamples in .txt, .csv, or .tsv format</i> </td>
  </tr>
  <tr>
   <td>labels_filename</td>
   <td>Name of the file that contains the labels of the datapoints</td>
   <td><i>None</i></td>
   <td>The labels must be in the form of an array with no labels</i> </td>
  </tr>
  <tr>
   <td>two_points_crossover_probability</td>
   <td>probability of the offsprings to be produced by the exchange of pieces from the parental individuals</td>
   <td>0.9 (90%)</td>
   <td>Control what percentage of offspring will be the result of the crossover of its parental chromosomes. The higher the probability, the less conservative the solutions are (dependig on the similarity of the parental solutions). If it is used along with the <code>arithmetic_crossover_probability</code> their sum should not be greater than 1. <i>The recommended values are [0.75-0.95]</i></td>
  </tr>
  <tr>
   <td>arithmetic_crossover_probability</td>
   <td>probability of an arithmetic crossover of the parental individuals to produce the offsprings</td>
   <td>0.0 (0%)</td>
   <td>Control what percentage of offspring will be the result of the crossover of its parental chromosomes. The higher the probability, the less conservative the solutions are (dependig on the similarity of the parental solutions). If it is used along with the <code>two_points_crossover_probability</code> their sum should not be greater than 1. <i>The recommended values are [0.0-0.1]</i></td>
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

<h3><i>(Comming soon)</i> Example of calling MEvA-X_V2 from terminal:</h3>

`test`

```
python MEvA-X.py -K 10 -P 50 -G 200 --dataset my_data.txt --labels my_labels.tsv -FS precalculated_features.csv --output_dir current_folder -cop 0.9 -acp 0 -mp 0.1 -goal_sig_lst 0.8 2 0.8 1 1 0.7 0.7 1 2 0.5 2
```
