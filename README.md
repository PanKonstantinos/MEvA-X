# MEvA-X
An open source biomarker discovery tool based on a multi-objective Evolutionary algorithm and the XGBoost Classifier.
Benchmarked on 2 datasets. One omics and one clinical. With MEvA-X the performance of the XGBoost Classifiers improved their overall performance and/or the simplicity of the final models. 

Dependencies:
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
    <td>pickle5</td>
    <td></td>
    <td>https://pypi.org/project/pickle5/</td>
  </tr>
  <tr>
    <td>knnimpute</td>
    <td></td>
    <td>https://github.com/iskandr/knnimpute</td>
  </tr>
  <tr>
    <td>requests</td>
    <td>>2.25</td>
    <td>https://pypi.org/project/requests/</td>
  </tr>
</table> 

Example for calling MEvA-X from terminal:
```python MEvA-X_V1.0.0.py```

(Comming soon) Example for calling MEvA-X_V2 from terminal:
```
python MEvA-X.py -K 10 -P 50 -G 200 --dataset my_data.txt --labels my_labels.tsv -FS precalculated_features.csv --output_dir current_folder -cop 0.9 -acp 0 -mp 0.1 -goal_sig_lst 0.8 2 0.8 1 1 0.7 0.7 1 2 0.5 2
```
