import argparse
import random
import math
import copy
import time
import sys
import multiprocessing as mp
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score, fbeta_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import manhattan_distances
from scipy.stats import ranksums
from sklearn.ensemble import VotingClassifier
import logging
import os
import mifs

logging.basicConfig(filename='LogFile.log',level=logging.DEBUG, filemode='w')

class Individual():
	"""
	Creates an Individual object.

	Attributes:
		individual: individual solution --> Just one individual not the whole individual matrix!
		process_i: i-th process
		dataset: the input dataset
		dataset_with_missing_values: the dataset with missing values imputated
		labels: (list) List with output labels per sample
		num_of_folds: (int) the number of folds
		output_folder: the output folder
	"""

	def __init__(self, individual, process_i, dataset, labels, num_of_folds, filter_mask, multiclass=True, random_state=None, last_eval=False, feature_names=None):
		self.individual = np.array(individual)
		self.process_i = process_i
		self.dataset = np.array(dataset)
		self.labels = labels
		self.num_of_folds = num_of_folds
		self.parameters = parameters #Global parameter
		self.filter_mask = filter_mask
		self.multiclass = multiclass
		self.random_state = random_state
		self.last_eval = last_eval
		self.feature_names = feature_names

	def evaluate(self):
		'''
		Evaluates the solutions of the population one by one (in parallel processes) by creating XGBoost models and evaluating their performances
		
		Input: an Individual class object
		Arduments: None
		Output: 
			- An array of the evaluation metrics and the standard deviation of the K-fold cross validation
			- An array of the FPR and TPR to build the AUC ROC
		'''
		logging.info(f'\n\n> Training models for individual # {self.process_i} ################################')
		evaluation_values = []
		l = self.individual.shape[0] - self.parameters

		logging.info(f'Selected_features == 1 {str(np.count_nonzero(self.individual[self.parameters:]==1))} features out of: {l} ')
		logging.info(f'Selected_features > 0.5 {str(np.count_nonzero(self.individual[self.parameters:]>0.5))} features out of {l} ')

		mask = self.filter_mask.copy()

		# No need to keep the parameters in the mask variable. We use it only for the features we want to select
		mask = mask[self.parameters:].round()
		mask = mask.astype(bool)#np.array(mask,dtype=bool)
		if mask.any():
			num_rounds = 1000
			xgb_params = {'eta':self.individual[4], 'max_depth':int(self.individual[5]),
						'gamma':self.individual[6], 'lambda':self.individual[7],
						'alpha':self.individual[8], 'min_child_weight':self.individual[9],
						'scale_pos_weight':self.individual[10],
						'nthread':2,'objective':'binary:logistic', 'eval_metric':['auc']}
						#'colsample_bytree':self.individual[11],'subsample':self.individual[12],\
			if self.multiclass==True:
				xgb_params['objective']='multi:softmax'
				xgb_params['eval_metric']=['mlogloss']
				xgb_params['num_class']=np.unique(self.labels).shape[0]


			inputs = self.dataset[mask,:] # Apply the mask on the dataset to keep ONLY the "approved" features as inputs
			inputs = inputs.T # Transpose the input to be Samples X Features
			outputs = self.labels.copy()


			N_trees=[]
			metrics_array = []#np.empty((10,self.num_of_folds))
			i=0
			fpr_tpr_array = []
			skf = StratifiedKFold(n_splits = self.num_of_folds, shuffle = True, random_state=self.random_state)
			for train_index, test_index in skf.split(inputs, outputs):
				training_inputs, testing_inputs = inputs[train_index], inputs[test_index]
				training_outputs, testing_outputs = outputs[train_index], outputs[test_index]
				dtrain = xgb.DMatrix(training_inputs, label = training_outputs)
				deval = xgb.DMatrix(testing_inputs, label = testing_outputs)
				metrics_array.append([])

				watchlist = [(deval,'eval')]

				verbose_eval = False
				booster = xgb.train(params = xgb_params, dtrain = dtrain, num_boost_round = num_rounds,
									evals = watchlist, early_stopping_rounds = 100,verbose_eval = verbose_eval)

                ############################################

				trees_strings = booster.get_dump(dump_format='text')
				total_splits = 0
				#total_cv_splits = 0
				for j,tree_string in enumerate(trees_strings):
					if j >= booster.best_ntree_limit:
						break
					n_nodes = len(tree_string.split('\n')) - 1
					n_leaves = tree_string.count('leaf')
					total_splits += n_nodes - n_leaves


				N_trees.append(booster.best_ntree_limit)
				predictions = booster.predict(deval, iteration_range = (0,booster.best_ntree_limit))+1e-08

				############################################

                ######### Calculating AUC,FPR,TPR  #########

				fpr, tpr, threshold = roc_curve(testing_outputs, predictions)
				roc_auc = auc(fpr, tpr)

				fpr_tpr_array.append((fpr, tpr, threshold, roc_auc))

                ############################################

				# Metrics calculation
				# Feature complexity
				metrics_array[i].append(10/(10+np.sum(mask))) #0 complexity features

				metrics_array[i].append(accuracy_score(deval.get_label(),predictions.round())) #1 Accuracy

				metrics_array[i].append(1-(total_splits/(booster.best_ntree_limit*(2**(xgb_params['max_depth'])-1)))) #2 complexity Splits

				metrics_array[i].append(weighted_geometric_mean(deval.get_label(),predictions.round(),np.unique(self.labels))) #3 wGM

				mertics_calc = mertics_calculator(deval.get_label(),predictions) #3

				metrics_array[i].append(mertics_calc['f1_score']) #4 F1 score
				metrics_array[i].append(mertics_calc['f2_score']) #5 F2 score
				metrics_array[i].append(mertics_calc['Precision']) #6 Precision
				metrics_array[i].append(mertics_calc['Recall']) #7 Recall
				metrics_array[i].append(mertics_calc['roc_auc']) #8 AUC
				metrics_array[i].append(mertics_calc['Balanced_accuracy']) #9 balanced Accuracy
				metrics_array[i].append(np.asarray(metrics_array[i]).mean()) #10 Overall score
				if self.multiclass:
					metrics_array[i].append(1/mertics_calc['manhattan_distance'] if mertics_calc['manhattan_distance']>0 else 1.0) #9
				i+=1

			#USE THE metrics_array TO CALCULATE STDV & MEAN!
			mean_array = np.array(metrics_array).mean(axis=0)
			stdv_array = np.array(metrics_array).std(axis=0)

			goal2 = [mean_array[1],stdv_array[1]] # accuracy/(float(self.num_of_folds)) # Avg accuracy
			avg_num_trees = mean_array[1] # total_num_trees/float(self.num_of_folds)

			goal3 = [mean_array[2],stdv_array[2]] # complexity metric
			goal4 = [mean_array[3],stdv_array[3]] # geo_mean/(float(self.num_of_folds)) # Avg geometric mean
			goal5 = [mean_array[4],stdv_array[4]] # f1_score/float(self.num_of_folds)
			goal6 = [mean_array[5],stdv_array[5]] # f2_score/float(self.num_of_folds)
			goal7 = [mean_array[6],stdv_array[6]] # precision/float(self.num_of_folds)
			goal8 = [mean_array[7],stdv_array[7]] # recall/float(self.num_of_folds)
			goal9 = [mean_array[8],stdv_array[8]] # roc_auc/float(self.num_of_folds)
			goal10 = [mean_array[9],stdv_array[9]] # Balanced Acc
			if self.multiclass:
				goal11 = [mean_array[9],stdv_array[9]]

			## Metric not used
			#goal12 = iba/float(self.num_of_folds)
			
			N_trees = int(np.median(N_trees))
		else:
			N_trees = 0
			mean_array = np.zeros(9) # 1.feature complexity, 2.Acc, 3.wGM, ..., 9. bal Acc
			stdv_array = np.zeros(9) # 1.feature complexity, 2.Acc, 3.wGM, ..., 9. bal Acc
			fpr_tpr_array = np.zeros((self.num_of_folds,4))
			goal2 = goal3 = goal4 = goal5 = goal6 = goal7 = goal8 = goal9 = goal10 = [0,0]
			if self.multiclass:
				goal11 = [0,0]
			## Metric not used
			#goal12 = 0
			
		evaluation_values.append(goal2[0]) # Goal 2 is the average accuracy
		evaluation_values.append(goal3[0])	# Goal 3 is (#of_samples-#of_avg_svs)/#of_samples --> [0,1]
		evaluation_values.append(goal4[0]) # Goal 4 is the average weighted geometric mean
		evaluation_values.append(goal5[0]) # Goal 5 is the average f1 score
		evaluation_values.append(goal6[0])	# Goal 6 is the average f2 score
		evaluation_values.append(goal7[0]) # Goal 7 is the average precision
		evaluation_values.append(goal8[0]) # Goal 8 is the average recall
		evaluation_values.append(goal9[0])	# Goal 9 is the average roc_auc
		evaluation_values.append(goal10[0]) # Goal 10 is the average Balanced accuracy
		if self.multiclass:
			evaluation_values.append(goal11[0]) # Goal 11 is the average manhattan distance
		evaluation_values.append(N_trees)
		## Metric not used
		#evaluation_values.append(goal12) # Goal 12 is the index of Balanced Accuracy [IBA]
		

		if self.process_i==0:
			print(f'Acc: {goal2}, Model_compl[splits]: {goal3}, wGM: {goal4}')
			print(f'F1: {goal5}, F2: {goal6}, Precision: {goal7}, Recall: {goal8}')
			print(f'Roc_auc: {goal9}, Balanced_accuracy: {goal10}')
			if self.multiclass:
				print(f'Manhattan_distance: {goal11}')
		return np.array(evaluation_values, dtype=float), list(zip(mean_array,stdv_array)), fpr_tpr_array # returns to evaluate_individuals

	def training_best(self,N_trees):
		'''
		Trains one final model based on the parameters of the individual in the first place ot the array (The one with the highest overall score)
		Attributes: 
			- Inherit attributes of the Individual class
			- N_trees (int) : Number of trees to use based on the optimization of the parameters for this solution
		Output: <XGBCalssifier> the trained model
		'''
		logging.info(f'\n\n> Training models for individual # {self.process_i} ################################')
		evaluation_values = []
		l = self.individual.shape[0] - self.parameters

		logging.info(f'Selected_features == 1 {str(np.count_nonzero(self.individual[self.parameters:]==1))} features out of: {l} ')
		logging.info(f'Selected_features > 0.5 {str(np.count_nonzero(self.individual[self.parameters:]>0.5))} features out of {l} ')

		mask = self.filter_mask.copy()

		# No need to keep the parameters in the mask variable. We use it only for the features we want to select
		mask = mask[self.parameters:].round()
		mask = np.array(mask,dtype=bool)
		assert mask.sum()==self.dataset.shape[1]

		num_rounds = N_trees
		xgb_params = {'eta':self.individual[4], 'max_depth':int(self.individual[5]),
				'gamma':self.individual[6], 'lambda':self.individual[7],
				'alpha':self.individual[8], 'min_child_weight':self.individual[9],
				'scale_pos_weight':self.individual[10],
				'nthread':mp.cpu_count(),'objective':'binary:logistic', 'eval_metric':['auc']}
		if self.multiclass==True:
			xgb_params['objective']='multi:softmax'
			xgb_params['eval_metric']=['mlogloss']
			xgb_params['num_class']=np.unique(self.labels).shape[0]


		inputs = self.dataset.copy()
		outputs = self.labels.copy()

		dtrain = xgb.DMatrix(inputs,label=outputs)
		verbose_eval = False
		model = xgb.train(params = xgb_params, dtrain = dtrain, num_boost_round = num_rounds, verbose_eval = verbose_eval)

		return model # pickle.dump(model, open(output_path+f'Pareto_1_results/Models/{eval_name}', "wb"))

	def prepare_models_majority_voting(self, N_trees):
		'''
		Parameters
		----------
		N_trees : int
			The maximum number of trees to train in the ensemble.

		Returns
		-------
		model : xgb.Booster
			Returns an xgboost Booster ready to train (but not trained yet) with the parameters of the individual of the First Pareto Frontier.
		'''
		logging.info(f'\n\n> Training models for individual # {self.process_i} ################################')
		evaluation_values = []
		l = self.individual.shape[0] - self.parameters

		mask = self.filter_mask.copy()
		# No need to keep the parameters in the mask variable. We use it only for the features we want to select
		mask = mask[self.parameters:].round()
		mask = np.array(mask,dtype=bool)
		if mask.shape[0]==None or mask.shape[0]==0 or mask.shape[0] is None:
			print(f'No features kept from the individual : {self.process_i}')
			return None

		num_rounds = N_trees
		xgb_params = {'eta':self.individual[4], 'max_depth':int(self.individual[5]),
				'gamma':self.individual[6], 'lambda':self.individual[7],
				'alpha':self.individual[8], 'min_child_weight':self.individual[9],
				'scale_pos_weight':self.individual[10],
				'nthread':mp.cpu_count(),'objective':'binary:logistic', 'eval_metric':['auc']}
				#'colsample_bytree':self.individual[11],'subsample':self.individual[12],\
		if self.multiclass==True:
			xgb_params['objective']='multi:softmax'
			xgb_params['eval_metric']=['mlogloss']
			xgb_params['num_class']=np.unique(self.labels).shape[0]

		inputs = self.dataset[mask,:] # Apply the mask on the dataset to keep ONLY the "approved" features as inputs
		inputs = inputs.T # Transpose the input to be Samples X Features
		active_feature_names = self.feature_names[mask]
		outputs = self.labels.copy()

		dtrain = xgb.DMatrix(inputs,label=outputs)
		verbose_eval = False
		model = xgb.Booster(xgb_params) # XGBClassifier(params)
# 		model = xgb.train(params = xgb_params, dtrain = dtrain, num_boost_round = num_rounds, verbose_eval = verbose_eval)

		return model

def majority_voting(individuals, dataset, labels, filter_mask, index, num_of_folds, N_trees, multiclass, output_folder, par_type):
	skf = StratifiedKFold(n_splits = num_of_folds, shuffle = True, random_state=np.random.randint(50))
	inputs = dataset.T.copy()
	fpr_array_soft = list()
	tpr_array_soft = list()
	auc_array_soft = list()
	fpr_array_hard = list()
	tpr_array_hard = list()
	auc_array_hard = list()
	fpr_tpr_array_soft = list()
	fpr_tpr_array_hard = list()
	metrics_array_soft = list()
	metrics_array_hard = list()
	i=0
	pareto1_intersected_features = filter_mask[index,parameters:].any(axis=0) #get the intersection of the feature names

	dict_of_k_fold_pred = {}
	for train_index, test_index in skf.split(inputs, labels):
		training_inputs, testing_inputs = inputs[train_index], inputs[test_index]
		training_outputs, testing_outputs = labels[train_index], labels[test_index]
        
		pareto_results_soft = list()
		pareto_results_hard = list()
		p1_fnames = list()

		mdls = list()
		prds = list()
		dict_of_preds = {}
		for ii,p1_indx in enumerate(index):
			feat_sel_indx = filter_mask[p1_indx,parameters:]

			deval = xgb.DMatrix(testing_inputs[:,feat_sel_indx], label = testing_outputs)
			mdl = Individual(individuals[p1_indx], p1_indx, training_inputs[:,feat_sel_indx],  training_outputs, num_of_folds, filter_mask[p1_indx], multiclass).training_best(N_trees[index][ii])
			mdls.append(mdl)

			n_train_iter = N_trees[index][ii]
			predictions = mdl.predict(deval, iteration_range = (0, int(n_train_iter)))+1e-08#(0, mdl.best_ntree_limit))
			dict_of_preds[f'solution_{ii}'] = predictions
            

			## If a solution classifies every sample in one class, skip this solution
			if np.unique(predictions.round()).shape[0]==1:
				continue

			feat_sel_names = feature_names[feat_sel_indx]
			p1_fnames.append(feat_sel_names.tolist())
			prds.append(predictions)

			with open(output_folder+'names_of_features.txt','a') as votes_file:
				votes_file.write(f'indiv {p1_indx}\n{feat_sel_names.tolist()}\n')
			with open(output_folder+'votes.txt','a') as votes_file:
				votes_file.write(f'indiv {p1_indx}\t{predictions}\n')
			with open(output_folder+'votes_hard.txt','a') as votes_file:
				votes_file.write(f'indiv {p1_indx}\t{predictions.round()}\n')
			#print(predictions)
			pareto_results_soft.append(predictions)
			pareto_results_hard.append(predictions.round())
        
		dict_of_k_fold_pred[f'fold_{i+1}'] = dict_of_preds
		metrics_array_soft.append([])
		metrics_array_hard.append([])
        
		pickle.dump(dict_of_k_fold_pred, open(output_folder+'Dictionary_of_predictions.pkl', "wb"))

		MJV_soft_predictions = np.asarray(pareto_results_soft).mean(axis=0)
		MJV_hard_predictions = np.asarray(pareto_results_hard).mean(axis=0)

		fpr, tpr, threshold = roc_curve(testing_outputs, MJV_soft_predictions)
		roc_auc = auc(fpr, tpr)
		fpr_tpr_array_soft.append((fpr, tpr, threshold, roc_auc))
		fpr_array_soft.append(fpr)
		tpr_array_soft.append(tpr)
		auc_array_soft.append(roc_auc)

		fpr, tpr, threshold = roc_curve(testing_outputs, MJV_hard_predictions)
		roc_auc = auc(fpr, tpr)
		fpr_tpr_array_hard.append((fpr, tpr, threshold, roc_auc))
		fpr_array_hard.append(fpr)
		tpr_array_hard.append(tpr)
		auc_array_hard.append(roc_auc)

		metrics_array_soft[i].append(accuracy_score(deval.get_label(),MJV_soft_predictions.round())) #1 Accuracy
		metrics_array_hard[i].append(accuracy_score(deval.get_label(),MJV_hard_predictions.round())) #1 Accuracy



		metrics_array_soft[i].append(weighted_geometric_mean(deval.get_label(),MJV_soft_predictions.round(),np.unique(labels))) #3 wGM
		metrics_array_hard[i].append(weighted_geometric_mean(deval.get_label(),MJV_hard_predictions.round(),np.unique(labels))) #3 wGM

		mertics_calc_soft = mertics_calculator(deval.get_label(),MJV_soft_predictions)
		mertics_calc_hard = mertics_calculator(deval.get_label(),MJV_hard_predictions)

		metrics_array_soft[i].append(mertics_calc_soft['f1_score']) #4 F1 score
		metrics_array_hard[i].append(mertics_calc_hard['f1_score']) #4 F1 score

		metrics_array_soft[i].append(mertics_calc_soft['f2_score']) #5 F2 score
		metrics_array_hard[i].append(mertics_calc_hard['f2_score']) #5 F2 score

		metrics_array_soft[i].append(mertics_calc_soft['Precision']) #6 Precision
		metrics_array_hard[i].append(mertics_calc_hard['Precision']) #6 Precision

		metrics_array_soft[i].append(mertics_calc_soft['Recall']) #7 Recall
		metrics_array_hard[i].append(mertics_calc_hard['Recall']) #7 Recall

		metrics_array_soft[i].append(mertics_calc_soft['roc_auc']) #8 AUC
		metrics_array_hard[i].append(mertics_calc_hard['roc_auc']) #8 AUC

		metrics_array_soft[i].append(mertics_calc_soft['Balanced_accuracy']) #9 balanced Accuracy
		metrics_array_hard[i].append(mertics_calc_hard['Balanced_accuracy']) #9 balanced Accuracy

		metrics_array_soft[i].append(np.asarray(metrics_array_soft[i]).mean()) #10 Overall score
		metrics_array_hard[i].append(np.asarray(metrics_array_hard[i]).mean()) #10 Overall score

		if multiclass:
			metrics_array_soft[i].append(1/mertics_calc_soft['manhattan_distance'] if mertics_calc_soft['manhattan_distance']>0 else 1.0) #9
			metrics_array_hard[i].append(1/mertics_calc_hard['manhattan_distance'] if mertics_calc_hard['manhattan_distance']>0 else 1.0) #9

		i+=1


	pd.DataFrame({'fpr':fpr_array_soft,'tpr':tpr_array_soft,'auc':auc_array_soft}).to_csv(f"{output_folder}fpr_tpr_DataFrame_soft_{par_type}.csv")
	pd.DataFrame({'fpr':fpr_array_hard,'tpr':tpr_array_hard,'auc':auc_array_hard}).to_csv(f"{output_folder}fpr_tpr_DataFrame_hard_{par_type}.csv")


	#USE THE metrics_array TO CALCULATE STDV & MEAN!
	mean_array_soft = np.array(metrics_array_soft).mean(axis=0)
	stdv_array_soft = np.array(metrics_array_soft).std(axis=0)
	mean_array_hard = np.array(metrics_array_hard).mean(axis=0)
	stdv_array_hard = np.array(metrics_array_hard).std(axis=0)
	names_list = ['Accuracy', 'wGM', 'F1', 'F2', 'Precision', 'Recall', 'AUC', 'balanced_Accuracy', 'Overall_score']

	print(par_type)
	print('The soft Majority vote returns:')
	for ii in range(mean_array_soft.shape[0]):
		print('%s %.3f ± %.3f\n' % (names_list[ii], mean_array_soft[ii], stdv_array_soft[ii]),end='\n\n')
	print('The hard Majority vote returns:')
	for ii in range(mean_array_hard.shape[0]):
		print('%s: %.3f ± %.3f\n' % (names_list[ii], mean_array_hard[ii], stdv_array_hard[ii]))


	soft = np.vstack([mean_array_soft, stdv_array_soft]).T
	hard = np.vstack([mean_array_hard, stdv_array_hard]).T
	majority_vote_df_soft = pd.DataFrame(soft,columns=["Mean","std"], index=names_list)
	majority_vote_df_hard = pd.DataFrame(hard,columns=["Mean","std"], index=names_list)
	majority_vote_df_soft.to_csv(f'{output_folder}Majority_Voting_soft_{par_type}.csv')
	majority_vote_df_hard.to_csv(f'{output_folder}Majority_Voting_hard_{par_type}.csv')
	#print(p1_fnames)
	print(np.unique(p1_fnames))
	print('Array of 1st Pareto feature names:\n',pareto1_intersected_features)
	return

def mertics_calculator(y_true, y_pred_proba, avg='weighted'):
	y_pred = y_pred_proba.round()
	scores = {}
	scores['Balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred) # bAccuracy = 1.0
	scores['f2_score'] = fbeta_score(y_true, y_pred, beta = 2.0, average=avg, zero_division = 0) # F2_weighted = 1.0
	scores['Precision'], scores['Recall'], scores['f1_score'], _ = precision_recall_fscore_support(y_true, y_pred, average=avg,zero_division = 0)
	scores['roc_auc'] = roc_auc_score(y_true, y_pred_proba, average = avg) # Roc_auc = 0
	scores['manhattan_distance'] = manhattan_distances(y_true.reshape(1,-1), y_pred.reshape(1,-1), sum_over_features=True).squeeze()# Manhattan Distance = 3.5
	''' Metric not used
	scores['IBA'] = IBA(dtrain = y_true, predt = y_pred)[1]
	'''
	return scores

def weighted_geometric_mean(y_true,y_pred,unique_labels=None):
	"""
	MCM[:,0,0] => TN
	MCM[:,1,0] => FN
	MCM[:,1,1] => TP
	MCM[:,0,1] => FP
	"""
	MCM = multilabel_confusion_matrix(y_true, y_pred, labels=unique_labels)
	GM = 0
	if MCM.shape[0]==2:
		# No need for support if we have only two classes (eg. 1/n*(sqrt()*5 + sqrt()*10) = 1/n*((5+10)*sqrt()) = n/n * sqrt())
		GM = math.sqrt((MCM[0,1,1] / (MCM[0,1,0] + MCM[0,1,1])) * (MCM[0,0,0] / (MCM[0,0,0] + MCM[0,0,1])))
	else:
		support_sum = 0
		for i in range(MCM.shape[0]):
			support = MCM[i,1,0]+MCM[i,1,1] # count only samples in the current class
			support_sum += support
			if (MCM[i,1,0] + MCM[i,1,1]) == 0 or (MCM[i,0,0] + MCM[i,0,1]) == 0:
				geo_mean = 0
			else:
				geo_mean = math.sqrt((MCM[i,1,1] / (MCM[i,1,0] + MCM[i,1,1])) * (MCM[i,0,0] / (MCM[i,0,0] + MCM[i,0,1])))

			GM += geo_mean*support
			print(f'Support for class_{i} = {support} and GM = {geo_mean}')
		GM = GM/support_sum
	return GM

def preprocessing_function(dataset_filename, labels_filename, as_pandas=False):
	delimeter = find_delimiter(dataset_filename)
	[dataset, feature_names, sample_names] = parsing_data_and_labels(dataset_filename = dataset_filename,
																	delimiter_dataset = delimeter, as_pandas = as_pandas)
	delimeter = find_delimiter_labels(labels_filename)
	labels = parsing_data_and_labels(labels_filename = labels_filename, delimiter_labels = delimeter, as_pandas = False)
	return dataset, feature_names, sample_names, labels


def find_delimiter(dataset_filename):
	"""
	Figures out which delimiter is being used in given dataset.

	Args:
		dataset_filename (string): the dataset filename

	Returns:
		(string): "," if CSV content, "\t" if TSV content.
	"""
	with open(dataset_filename, 'r') as handle:
		head = next(handle)
		head = next(handle)
	if "," and "\t" in head:
		return "\t"	# The case where the comma is the decimal separator (greek system)
	elif "\t" in head:
		return "\t"
	elif "," in head:
		return ","

def find_delimiter_labels(labels_filename):
	"""
	Figures out which delimiter is being used in given labels dataset.

	Args:
		labels_filename (string): the labels filename

	Returns:
		(string): "," if CSV content, "\t" if TSV content.
	"""
	with open(labels_filename, 'r') as handle:
		head = next(handle)
	if "," and "\t" in head:
		return "\t"	# The case where the comma is the decimal separator (greek system)
	elif "\t" in head:
		return "\t"
	elif "," in head:
		return ","

def parsing_data_and_labels(dataset_filename=None, delimiter_dataset=None,\
                            labels_filename=None, delimiter_labels=None, as_pandas=False):
	return_dataset = False
	return_labels = False
	'''Dataset'''
	if dataset_filename and delimiter_dataset:
		return_dataset = True
		dataset = pd.read_csv(dataset_filename, sep = delimiter_dataset, header = 0, index_col = 0, na_values = [-1000,-999,999,' ','_','-',''])
		dataset.columns = dataset.columns.str.strip() #get rid of spaces at the front and at the end of the column names
		dataset.index = dataset.index.str.strip()
		dataset.columns = dataset.columns.str.replace(' ','_') #get rid of spaces and replace them with underscore '_' in the midle of the column names
		dataset.index = dataset.index.str.replace(' ','_')
		[dataset, feature_names, sample_names] = mean_duplicated(dataset)
		if as_pandas==False:
			dataset = dataset.to_numpy() # Returns just the values of the dataset in a numpy array
	'''Labels'''
	if labels_filename and delimiter_labels:
		return_labels = True
		labels = pd.read_csv(labels_filename, sep = delimiter_labels, header = None, na_values = [-1000,-999,999,' ','_','-',''])
		#if as_pandas==False:
		labels = labels.to_numpy().flatten()
	'''Return'''
	if return_dataset and return_labels:
		return dataset, feature_names, sample_names, labels
	elif return_dataset==True and return_labels==False:
		return dataset, feature_names, sample_names
	elif return_dataset==False and return_labels==True:
		return labels

def mean_duplicated(dataset):
	"""
	mean_dublicated method, finds duplicated index names and means out the values by sample (column-wise).
	Also returns the feature names and sample names

	Inputs:
		- dataset Features(rows) X Samples(columns)
	Returns:
		- dataset (without dublicated feature names)
		- data_index_names (Unique feature names)
		- data_col_names (Sample names)
	"""
	dataset = dataset.groupby(dataset.index, sort=False).mean()
	data_col_names = dataset.columns.values
	data_index_names = dataset.index.values

	return dataset, data_index_names, data_col_names


def impute(labels,data,imputer='knn'):
	imputed_dataset=data.copy()
	for feat in data.index:
		if any(data.loc[feat,:].isnull()):
			if imputer=='knn':
				imputed_dataset = knnimputer(data)
			else:
				for lab in np.unique(labels):
					#if any(data.loc[feat,:].isnull()):
					if data.loc[feat,np.asarray(labels==lab).squeeze()].shape[0]<=1:
						imputed_dataset.loc[feat,np.asarray(labels==lab).squeeze()]=data.loc[feat,np.asarray(labels==lab).squeeze()].fillna(data.loc[feat,:].mean())
					else:
						imputed_dataset.loc[feat,np.asarray(labels==lab).squeeze()]=data.loc[feat,np.asarray(labels==lab).squeeze()].fillna(data.loc[feat,np.asarray(labels==lab).squeeze()].mean())
						if any(imputed_dataset.loc[feat,np.asarray(labels==lab).squeeze()].isnull()):
							imputed_dataset.loc[feat,np.asarray(labels==lab).squeeze()]=data.loc[feat,np.asarray(labels==lab).squeeze()].fillna(data.loc[feat,:].mean())
	return imputed_dataset

def knnimputer(dataset_initial):
	from knnimpute import knn_impute_optimistic

	dataset_initial = dataset_initial.T # in order to have Samples X Features
	imp_dataset = knn_impute_optimistic(np.asarray(dataset_initial), np.isnan(np.asarray(dataset_initial)), k=5)
	return imp_dataset.T # in order to have Features X Samples as the original dataset

def normalize_dataset(dataset, output_folder=None,min_max_scaler=True):
	if min_max_scaler:
		normalizer = MinMaxScaler()
		dataset_normalized = normalizer.fit_transform(dataset.T)
		dataset_normalized = dataset_normalized.T
	else:
		normalizer = MaxAbsScaler()
		dataset_normalized = normalizer.fit_transform(dataset.T)
		dataset_normalized = dataset_normalized.T
	return dataset_normalized

def transform_labels_to_numeric(labels,unique_labels):
	if isinstance(labels[0],str):
		encoder = LabelEncoder()
		encoder.fit(unique_labels)
		try:
			labels = encoder.transform(labels.T)
		except:
			labels = encoder.transform(labels)
		unique_labels = encoder.transform(unique_labels)
	return labels, unique_labels

def pareto_frontiers(evaluation_values):
	assigned = 0  # How many individuals have assigned to a frontier list so far
	population = evaluation_values.shape[1]
	fronts = np.zeros(population, dtype='int')
	front = 1
	eval_temp = evaluation_values.copy()

	logging.info("evaluation_values")
	logging.info(evaluation_values)

	while assigned < population:

		non_dominated_solutions = np.zeros(population, dtype=int)
		ordered_list = np.flip(np.argsort(eval_temp[0]))
		non_dominated_solutions[0] = ordered_list[0]
		number_of_non_dominated_solutions = 1

		for i in range(1, population):
			n = 0
			condition = True
			while n < number_of_non_dominated_solutions:
				solution1 = eval_temp[:-1,ordered_list[i]] # This is the list of all but the NON-dominated solutions
				solution2 = eval_temp[:-1,non_dominated_solutions[n]] # This is the list of NON-dominated solutions
				check = dominant_solution(solution1, solution2)
				if check==1: #New is dominant
					if number_of_non_dominated_solutions==1:
						non_dominated_solutions[0]=ordered_list[i]
						condition = False # Do NOT add a new non dominated solution, just put the new one on the first position
						break # STOP checking. The old solution is defeated and replaced
					else:
						number_of_non_dominated_solutions=number_of_non_dominated_solutions-1
						non_dominated_solutions = np.delete(non_dominated_solutions,n)
						continue # Continue checking the new solution compared to all other non dominated
				elif check==3: #Non-diminated is dominant
					condition = False # Do NOT add a new non-dominated solution
					break # STOP checking. The new solution is already defeated
				n += 1

			if condition:
				non_dominated_solutions[number_of_non_dominated_solutions]=ordered_list[i]
				number_of_non_dominated_solutions += 1
		sorted_non_dominated_solutions=sorted(non_dominated_solutions, reverse=True)
		assigned += number_of_non_dominated_solutions
		fronts[sorted_non_dominated_solutions[:number_of_non_dominated_solutions]] = front
		front += 1
		eval_temp[:,sorted_non_dominated_solutions[:number_of_non_dominated_solutions]] = -1
	return fronts


def dominant_solution(solution1, solution2):
    """
	Check = 2 is returned when there is no dominant solution between the two
	Check = 1 is returned when the 'new' solution dominates the 'old' one
	check = 3 is returned when the 'old'/non_dominated dominates the 'new' one
	"""
    check = 2
    if all(solution1 >= solution2) and any(solution1 > solution2):
        check = 1
    elif all(solution1 <= solution2) and any(solution1 < solution2):
        check = 3
    return check

def initialize_individuals(min_values, max_values, population, exessive_population=False):
	"""
	For the alternative method, check the script "Alternative initialization method.py"

	Remember to use the exessive_population in order to use Selection_tournament and then reach population!
	"""
	individuals = np.zeros((population,min_values.shape[0]))
	num_of_potential_inputs = min_values.shape[0] - parameters
	max_inp = 30 # @10/03/2021
	if max_inp > num_of_potential_inputs:
		max_inp = num_of_potential_inputs
	for i in range(population):
			num_of_selected_inputs = np.random.randint(1,max_inp+1)#int(np.ceil(np.random.uniform(0,max_inp)))
			for j in range(parameters):
					individuals[i,j] = np.random.uniform(min_values[j],max_values[j])
			selected_inputs = np.random.choice(a=num_of_potential_inputs, size=num_of_selected_inputs, replace=False)
			individuals[i,parameters+selected_inputs] = 1
	return individuals

def features_selected(out_vars):
	index = 0
	output_list = [[f'Ind_{out_vars[0,0]}: ',out_vars[1,0]]]
	temp_var = out_vars[0,0]
	for i in range(1,len(out_vars[0])):
		if out_vars[0,i] == temp_var:
			output_list[index].append(out_vars[1,i])
		else:
			output_list.append([f'Ind_{out_vars[0,i]}: ',out_vars[1,i]])
			index+=1
			temp_var = out_vars[0,i]
	return output_list

def evaluate_individual(individual):
	"""
	Acts the evaluate() method on an individual object.

	Args:
		individual: an object

	Returns: individual.evaluate().
	"""
	return individual.evaluate()


def evaluate_individuals(dataset, labels, individuals, goal_significances, num_of_folds, classification_problems,
						output_folder, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes, multiclass=True):

	filter_mask = filter_function(individuals, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes)

	# Create Individual Class instances and use them to find the fitness of the models. Then drop them to save memory
	eval_time_start = time.time()
	random_state = np.random.randint(500)
	results = []
	mean_std_all = []
	roc_auc_all = []
	for i, individual in enumerate(individuals):
		res,mean_std, roc_auc_indiv = evaluate_individual(Individual(individual, i, dataset, labels, num_of_folds, filter_mask[i], multiclass, random_state))
		results.append(res)
		mean_std_all.append(mean_std)
		roc_auc_all.append(roc_auc_indiv)
	'''In case I want to save some data from the Individuals, I have to:
	1. modify the return of the method evaluate_individual (e.g. return the stdv)
	2. split the results in lists
	'''

	eval_time_stop = time.time()
	print(f'Time to run CV: {eval_time_stop-eval_time_start}')
	with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Time to run CV: {eval_time_stop-eval_time_start}\n')

	# Convert the results to numpy array for easier handling
	results = np.array(results, dtype = float)
	## last element is the mean number of trees in the ensemble. I need it only for the final training
	N_trees = results[:,-1] # N_trees is the num_boost_round for the final models
	N_trees = N_trees.astype('int32')
	results = results[:,:-1] # drop the average number of trees from evaluation values
	print(f'N_trees = {N_trees}')
	# Creating the matrix that holds the evaluation values. It has +2 positions to hold 1. the feature complexity & 2. The overall score
	evaluation_values = np.empty([len(results[0])+2, individuals.shape[0]], dtype = float) # [[goal1],[goal2],[goal3],...[goal_n],[overall_score]]

	# Find the Model_complexity based on the #active_features in each individual solution
	for ind in range(individuals.shape[0]):
		''' HERE WE CHECK IF THE PARAMETERS ARE RIGHT TO FILTER THE GENES (Wilcoxon, mifs, SKB)'''

		number_of_selected_feature=0
		number_of_selected_feature = np.count_nonzero(filter_mask[ind,parameters:])
		goal1 = 0 # in case No_of_selectd_features <= 0
		if number_of_selected_feature > 0:
			#goal1 = 1/(1 + number_of_selected_feature) # 1 selected_feature = 0.5, 2 selected_features = 0.333, 9 selected_features = 0.1
			goal1 = 10/(10 + number_of_selected_feature) # 1 feature = 1, 2 features = 0.8333, ... 9 features = 0.4
		evaluation_values[0,ind]= goal1 # [[goal1.0,goal1.1,goal1.2,goal1.3,...,goal1.len(individuals)-1],[],[],[]]
	logging.info("Initial Feature Selection Completed Succesfully!")

	evaluation_values[1:-1] = results.T # Transpose the results to have a matrix: Eval_val X Indiv

	# Eval_values_overall = [(b0*x0)+(b1*x1)+(b2*x2)+...+(b_n-2*x_n-2)+(b_n-1*x_n-1)]
	for i in range(evaluation_values.shape[1]): # For all individuals
		evaluation_values[-1,i] = np.multiply(evaluation_values[:-1,i],goal_significances.T).mean()

	eval_metrics_names = ['Model_complexity #features','Accuracy','Model_complexity #splits',
	'weighted Geometric Mean','F1 score','F2 score','Precision','Recall','AUrocC',
	'Balanced_accuracy','Manhattan distance^-1','Overall score']
	for i in range(evaluation_values.shape[1]):
		logging.info('-----------------Individual:'+str(i)+'----------------------------')
		logging.info('--------------------------------------------------------------')
		for j in range(evaluation_values.shape[0]):
			if eval_metrics_names[j]=='Manhattan distance^-1' and multiclass==False:
				logging.info(f'| Goal_{j}\t|  {eval_metrics_names[j]}\t|\tNot used in Binary Classification problems')
				logging.info(f'| Goal_{j+1}\t|  {eval_metrics_names[j+1]}\t|\t{evaluation_values[j,i]}')
			else:
				logging.info(f'| Goal_{j}\t|  {eval_metrics_names[j]}\t|\t{evaluation_values[j,i]}')
		logging.info('--------------------------------------------------------------')
		# G1:Feature_compl, G2:Acc, G3:Sample_compl, G4:wGM, G5:F1, G6:F2, G7:Precision, G8:Recall, G9:AUC, G10:Manhatan_dist, G11:GM
	return evaluation_values, mean_std_all, roc_auc_all, N_trees

def create_different_classification_problems(labels, unique_labels):
	"""
	Creates different classification problems to input in the evaluate_individuals() function.

	Args:
		labels: (list) the labels

	Returns:
		classification_problems: (list) the classification problems
	"""

	classification_problems = np.zeros((math.factorial(len(unique_labels))//(math.factorial(len(unique_labels)-2)*2),labels.shape[0])) # n!/(k!*(n-k)!) -> n choose k
	number_of_classification_problems=0

	for i in range(len(unique_labels)):
		for k in range(i+1, len(unique_labels)):
			for j in range(labels.shape[0]):
				if labels[j]==unique_labels[i]:
					classification_problems[number_of_classification_problems,j] = 1
				elif labels[j]==unique_labels[k]:
					classification_problems[number_of_classification_problems,j] = -1
			number_of_classification_problems += 1
	for i in range(len(classification_problems)):
		#print(len(classification_problems[i]))
		logging.info(len(classification_problems[i]))
	logging.info("Classification problems were created successfully!")
	return classification_problems

def Wilcoxon_ranksums(dataset_normalized, classification_problems):
	'''l=len(individuals[ind])-parameters'''
	print('Calculating Wilcoxon ranksums')
	try:
		dataset_normalized = dataset_normalized.values.copy()
	except:
		print('Dataset_normalized is not Pandas.DataFrame')
	l = dataset_normalized.shape[0]
	selected = np.zeros(l)
	for feature in range(l):
		for i in range(len(classification_problems)):
			if selected[feature]==1:
				break

			data1=list()
			data2=list()
			for j in range(len(dataset_normalized[0])):
				if int(classification_problems[i,j])==1:
					if (float(dataset_normalized[feature,j])!=-1000):
						data1.append(dataset_normalized[feature,j])
				elif int(classification_problems[i,j])==-1:
					if (float(dataset_normalized[feature,j])!=-1000):
						data2.append(dataset_normalized[feature,j])
			if len(data1)>1 and len(data2)>1:
				[z,pvalue] = ranksums(data1,data2)
				if pvalue<0.05:
					selected[feature]=selected[feature]+1
	return selected

def mifs_calc(data, labels, method = 'JMI', n_features = 100, k_vals = 4):
	'''This Method is not functional yet!'''

	# FS_methods = mifs_calc(dataset, labels, k_vals=k_vals, n_features=n_features)
	# JMI_genes = FS_methods[:6]
	# mRMR_genes = FS_methods[7:]
	# Wilcoxon_genes = Wilcoxon_ranksums(dataset, classification_problems)
	# SelKBest_genes = SelectKBest(mutual_info_classif, k=100).fit_transform(dataset.T,labels).get_support(indices = True)

	if k_vals==None or k_vals==[]:
		start = 4
		finish = 10
		k_vals = list(range(start,finish+1))#[4,5,6,7,8,9,10]
	elif not(isinstance(k_vals, int)) and not(isinstance(k_vals, list)):
		raise TypeError('k must be an integer')


	if not (method in ['JMI','MRMR','JMIM']):#!='JMI' or method!='JMIM' or method!='mRMR':
		raise ValueError('method should be: \'JMI\', \'JMIM\' or \'MRMR\'')

	if not(isinstance(method,list)):
		method = [method]

	try:
		FS_methods = np.empty((len(k_vals)*len(method),n_features))
	except:
		FS_methods = np.empty((k_vals*len(method),n_features))
	i = 0
	if not(isinstance(k_vals,list)):
		k_vals = [k_vals]
	for meth in method:
		for k in k_vals:
			print(f'Calculating the Mutual Information with method: {meth} and {k}-NN')
			mif = mifs.MutualInformationFeatureSelector(meth,k=k,n_jobs=mp.cpu_count()-1,verbose=1, n_features=n_features)

			results = mif.fit(data.T,labels)
			rank_results = np.array(results.ranking_)
			FS_methods[i] = pd.Series(rank_results).T
			i+=1
	print(f'The shape is:{FS_methods.shape} and Transposed is {FS_methods.T.shape}')
	return FS_methods

def limit(num, minn=0, maxn=1):
	'''
	Constrains the value of a variable between minimum and maximum

	num: the variable we want to constrain
	minn: the minimum value num is allowed to have
	maxn: the maximum value num is allowed to have
	'''
	return max(min(num,maxn),minn)

def mutation(selected_individuals, population, min_values, max_values, mutation_probability, generations, current_gen, mu=0, s=0.1):
	'''
	Mutates the individual solutions by adding/subtracting to the parameters and by Adding/Removing genes
	'''
	if current_gen>=generations/3 and current_gen<generations/6:
		mutation_probability=mutation_probability/2
	elif current_gen>=generations/6:
		mutation_probability=mutation_probability/3

	for indiv in range(1, population):

		#Mutation on Parameter-genes ONLY
		for param in range(parameters):
			random_number = np.random.uniform(0,1)
			if random_number<mutation_probability:
				selected_individuals[indiv,param] += np.random.normal(mu, s*(max_values[param]-min_values[param]))
				# Correcting values that are out of boundaries
				selected_individuals[indiv,param] = limit(selected_individuals[indiv,param], min_values[param], max_values[param]-1e-05) #-1E-5 in order to avoid rounding to max value when we use int()

		## Mutation on Feature-genes Only
		off_genes = selected_individuals[indiv,parameters:]<=0.5
		on_genes = ~off_genes

		## Turn Genes ON
		added_features_flag = np.random.choice([True,False])#randint(0,2)
		if added_features_flag:
			#Add a new feature in this chromosome
			num_of_added_features = np.random.randint(1,6)

			positions=np.random.choice(np.arange(selected_individuals[indiv].shape[0]-parameters)[off_genes],\
										size=num_of_added_features, replace=False)
			selected_individuals[indiv,parameters+positions] = 1

		## Turn Genes OFF
		elif any(on_genes):#.sum()>1:
			if (on_genes).sum()<6:
				num_of_removed_features = limit(np.random.randint(0,on_genes.sum()), 1, on_genes.sum())
			else:
				num_of_removed_features = np.random.randint(1,6)

			non_zero_features_indx = np.arange(selected_individuals[indiv].shape[0]-parameters)[on_genes]
			if non_zero_features_indx.size>0:
				positions = np.random.choice(non_zero_features_indx, size=num_of_removed_features, replace=False)
				#print(selected_individuals[indiv,parameters+positions])
			else:
				positions = np.random.choice(selected_individuals[indiv].shape[0]-parameters, size=num_of_removed_features, replace=False)
			selected_individuals[indiv,parameters+positions] = 0

	return selected_individuals

def filter_function(individuals, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes):
	'''
	Applies the Feature selection method (if applicable) based on the parameter genes each individual has
	Parameters:
	-----------
		individuals [array]: The array of individual solutions with all the parameters for each one of them
		JMI_genes [array]: The array of positions of genes selected by the JMI method for filtering
		Wilcoxon_genes [array]: The array of positions of genes selected by the mRMR method for filtering
		mRMR_genes [array]: The array of positions of genes selected by the JMI method for filtering
		SelKBest_genes [array]: The array of positions of genes selected by the SelKBest method for filtering
	Return:
		filter_mask [array]: A boolean array of index for genes to turn ON/OFF
	-----------
	
	'''
	filter_mask = individuals.copy()
	for indiv in range(individuals.shape[0]):
		if int(individuals[indiv,0]) != 0:#No Feature Selection method
			if int(individuals[indiv,1]) < 2:

				if int(individuals[indiv,0]) == 1: #JMI
					if JMI_genes is None:
						filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()
						print('JMI FS has selected no features. No filtering will apply on the features!')
						continue
					else:
						FS_genes = JMI_genes[int(individuals[indiv,2])-4] # k = [4,5,6,...,10] that's why we subtract 4 to have the right index

				elif int(individuals[indiv,0]) == 2: #Wilcoxon ranksums
					if Wilcoxon_genes is None:
						filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()
						print('Wilcoxon rank sums FS has selected no features. No filtering will apply on the features!')
						continue
					else:
						#FS_genes = Wilcoxon_genes.to_numpy()[0]
						FS_genes = Wilcoxon_genes

				elif int(individuals[indiv,0]) == 3: #mRMR
					if mRMR_genes is None:
						filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()
						print('mRMR FS has selected no features. No filtering will apply on the features!')
						continue
					else:
						#FS_genes = mRMR_genes.iloc[int(individuals[indiv,2]-4)].to_numpy() # k = [4,5,6,...,10] that's why we subtract 4 to have the right index
						FS_genes = mRMR_genes[int(individuals[indiv,2]-4)]

				elif int(individuals[indiv,0]) == 4: #SelectKBest
					if SelKBest_genes is None:
						filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()
						print('Select K Best FS has selected no features. No filtering will apply on the features!')
						continue
					else:
						up_to = int(individuals[indiv,3])
						#FS_genes = SelKBest_genes.iloc[:,:up_to].to_numpy()[0]
						if isinstance(SelKBest_genes,np.ndarray):
							FS_genes = SelKBest_genes[:up_to]
						else:
							FS_genes = SelKBest_genes.scores_[np.flip(np.argsort(SelKBest_genes.scores_))[:up_to]]
			else:
				filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()
				continue

			if int(individuals[indiv,1]) == 0: # Feature Selection filter only
				temp_chrom = np.zeros(individuals.shape[1]-parameters)
				temp_chrom[FS_genes] = individuals[indiv,FS_genes+parameters]
				temp_chrom[FS_genes] = temp_chrom[FS_genes].round()
				filter_mask[indiv,parameters:] = temp_chrom

			elif int(individuals[indiv,1]) == 1: # Genetic Algorithm filter only -> Removes genes that the FS method found important
				temp_chrom = individuals[indiv,parameters:].copy()
				temp_chrom[FS_genes] = 0
				temp_chrom = temp_chrom.round()
				filter_mask[indiv,parameters:] = temp_chrom
		else:
			filter_mask[indiv,parameters:] = filter_mask[indiv,parameters:].round()

	return filter_mask.astype(bool)

def similarity_function(fronts, evaluation_values, individuals, sigma_share, max_values, min_values):
	'''
	Calculates and gegredes the solutions based on their similarity
	Parameters:
	-----------
		fronts [array]: Indicated the Pareto frontier for every individual solution
		evaluation_values [array]: The array of the evaluation values for each solution
		individuals [array]: The array of individual solutions with all the parameters for each one of them
		sigma_share [float]: The radius of the neighborhood in which solutions are considered similar
		max_values [array]: maximum value allowed for every position in the individual[i] array
		min_values [array]: minimum value allowed for every position in the individual[i] array
	Return:
		filter_mask [array]: A boolean array of index for genes to turn ON/OFF
	-----------
	
	'''
	evaluation_values_f = evaluation_values.copy()
	# evaluation_values_f_test = evaluation_values.copy()
	ms = np.empty(np.unique(fronts).shape[0])
	i=0
	for front in sorted(np.unique(fronts)):
		logging.info(f'Frontier: {str(front)}')
		ind = np.array(np.where(fronts == front)).ravel()

		dist_mat = np.zeros([ind.shape[0],ind.shape[0]],dtype = 'float') #  distance matrix of individuals belonging in the same frontier

		front_max_eval = np.max(evaluation_values_f[:-1,ind], axis = 1)
		with open(output_folder+'Pareto_highest_values.txt','a') as front_max_file:
			front_max_file.write(f'Pareto: {front} \n {front_max_eval}\n\n')

		for j in range(ind.shape[0]): # 0,1,2,3,4 (e.g. len(ind) = 5)
			m = 0
			for k in range(j,ind.shape[0]): # 0,1,2,3,4 | 1,2,3,4 | 2,3,4 | 3,4 | 4=X
				d = 0
				if j!=k:
					# for gene in range(individuals.shape[1]):
					for gene in range(parameters):
						d += ((individuals[ind[j],gene]-individuals[ind[k],gene])/float(max_values[gene]-min_values[gene]))**2
					d += ((individuals[ind[j],parameters:]-individuals[ind[k],parameters:])**2).sum()
					d = math.sqrt(d/individuals.shape[1])
					dist_mat[j,k] = dist_mat[k,j] = d
					if d<=sigma_share:
						m += 1-((d/sigma_share)**2)
				else:
					m+=1
			for p in range(j): # This is the lower-left part of the distance matrix we need to take into consideration for the m
				if p!=j and dist_mat[j,p]<=sigma_share:
					m += (1-((dist_mat[j,p]/float(sigma_share))**2))

			if m==0:
				m=1

			evaluation_values_f[:-1,ind[j]] = front_max_eval/m

			logging.info("m = "+str(m))
			ms[i] += m
		ms[i] /= ind.shape[0]
		print(f'Mean m for frontier {front} ({ind.shape[0]}) = {ms[i]}')
		i+=1
	return evaluation_values_f

def similarity_function_rounded(fronts, evaluation_values, individuals, sigma_share, max_values, min_values):
	'''
	Calculates the distance between individuals in the same Pareto
	'''
	evaluation_values_f = evaluation_values.copy()
	individuals_temp = individuals[:,parameters:].copy()
	individuals_temp = individuals_temp.round()
	ms = np.empty(np.unique(fronts).shape[0])
	i=0
	for i,front in enumerate(sorted(np.unique(fronts))):
		logging.info(f'Frontier: {str(front)}')

		ind = np.argwhere(fronts == front).ravel()

		dist_mat = np.zeros([ind.shape[0],ind.shape[0]],dtype = 'float') #  distance matrix of individuals belong in the same frontier

		front_max_eval = np.max(evaluation_values_f[:-1,ind], axis = 1)
		with open(output_folder+'Pareto_highest_values.txt','a') as front_max_file:
			front_max_file.write(f'Pareto: {front} \n {front_max_eval}\n\n')

		for j in range(ind.shape[0]): # 0,1,2,3,4 (e.g. len(ind) = 5)
			m = 0
			for k in range(j,ind.shape[0]): # 0,1,2,3,4 | 1,2,3,4 | 2,3,4 | 3,4 | 4=X
				d = 0 # distance of genes
				d_par = 0 # distance for parameters
				if j!=k:
					for gene in range(parameters):
						d_par += ((individuals[ind[j],gene]-individuals[ind[k],gene])/float(max_values[gene]-min_values[gene]))**2
					d_par = math.sqrt(d_par/parameters)
					#d += np.logical_xor(individuals_temp[ind[j]],individuals_temp[ind[k]]).sum()
					#d = math.sqrt(d/individuals.shape[1])
					d += np.logical_xor(individuals_temp[ind[j]],individuals_temp[ind[k]]).sum() # XOR(gene_a,i,gene_b,i)
					n_union = np.union1d(np.argwhere(individuals_temp[ind[j]]==1),np.argwhere(individuals_temp[ind[k]]==1)).shape[0]# #N_union
					if n_union > 0: #there is at least 1 gene active in any of the chromosomes
						d /= n_union

					d += d_par
					d /= 2.0

					dist_mat[j,k] = dist_mat[k,j] = d
					if d<=sigma_share:
						m += 1-((d/sigma_share)**2)
				else:
					m+=1
			for p in range(j): # This is the lower-left part of the distance matrix we need to take into consideration for the m
				if p!=j and dist_mat[j,p]<=sigma_share:
					m += (1-((dist_mat[j,p]/float(sigma_share))**2))

			if m==0:
				m=1

			evaluation_values_f[:-1,ind[j]] = front_max_eval/m
			logging.info("m="+str(m))
			ms[i] += m
		ms[i] /= ind.shape[0]
		print(f'Mean m for frontier {front} ({ind.shape[0]}) = {ms[i]}')
	return evaluation_values_f



def apply_evolutionary_process(generations, population, max_values, min_values, two_points_crossover_probability,
								arithmetic_crossover_probability, mutation_probability,dataset,
								labels, individuals, goal_significances, num_of_folds, classification_problems,  output_folder,
								JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes, eval_names, multiclass=True):

	max_eval_per_generation = np.empty(generations)
	average_eval_per_generation = np.empty(generations)
	sum_ranked_eval_per_generation = np.empty(generations)
	selected_individuals = np.empty((population, min_values.shape[0]))


	for rep in range(generations):
		logging.info("Generation:"+str(rep))
		print(f'Generation {rep}')


		out_vars = []#dummy_variable = []
		for gene in individuals:
			out_vars.append(np.argwhere(gene[parameters:]>0.5).ravel())
		print(f'Individual_0 has the following features selected: {out_vars[0]}')
		logging.info(f'The features used are:\n{out_vars}')


		with open(output_folder + 'Selected Features.txt','a') as feature_position_file:
			feature_position_file.write(f'Generation {rep}:\n')
			for i in range(len(out_vars)):
				if out_vars[i].shape[0]==0:
					feature_position_file.write(f'Indiv_{i} -> None\n')
				else:
					feature_position_file.write(f'Indiv_{i} -> {out_vars[i]}\n')
			feature_position_file.write('-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\n\n')


		#evaluate the population of solutions
		evaluation_values, mean_std_list, roc_auc_list, _ = evaluate_individuals(dataset, labels, individuals,
																		   goal_significances, num_of_folds, classification_problems,
																		   output_folder, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes, multiclass)

		"""
		evaluation_values:
			Goals X Individuals -> ~10-15 goals X population
		each column is an Individual
		evaluation_values:
		[[0.5, 		0.3333, 	0.1111, 	0.2, 		0.125],		->	1/(#selected_features+1)
		[33.3333, 	41.6666, 	39.5833, 	43.75, 		41.666],	->	Avg(accuracy)
		[0.4133, 	0.36, 		0.39333, 	0.34, 		0.38],		->	(#samples-#svms)/#samples
		[10.1826, 	12.6386, 	11.9758, 	13.232, 	12.601]]	->	((w0*1/(#selected_features+1))+(w1*avg_accuracy)+(w2*(#samples-#svms)/#samples))/3

		"""
		evaluation_values = np.array(evaluation_values)

		#Keep "Best" and average metrics for each generation in files

		#find and write to file maximum and average performances

		best_overall_indiv_pos = evaluation_values[-1].argmax() # Get the index of the individual with the highest overall score
		best_auc_indiv_pos = evaluation_values[-3].argmax()
		best_bAcc_indiv_pos = evaluation_values[-2].argmax()
		best_indiv_pos = best_overall_indiv_pos
		try:
			max_eval = evaluation_values[-1][best_indiv_pos[0]]
		except:
			max_eval = evaluation_values[-1][best_indiv_pos]

		with open(output_folder+'Index_of_best_individuals_per_metric.txt','a') as f:
			f.write(f'Gen{rep}:\nOverall: {best_overall_indiv_pos}\tAUC: {best_auc_indiv_pos}\tbAcc: {best_bAcc_indiv_pos}\n\n')

		max_eval_per_generation[rep]=max_eval
		with open(output_folder+"CV_best_performance.txt",'a') as bst_perf:
			bst_perf.write(str(max_eval_per_generation[rep])+"\n")

		average_eval = evaluation_values[:-1].mean()
		average_eval_per_generation[rep] = average_eval
		with open(output_folder+"CV_average_performance.txt",'a') as avg_perf:
			avg_perf.write(str(average_eval_per_generation[rep])+"\n")
		#average_performance_fid.write(str(average_eval_per_generation[rep])+"\n")

		with open(output_folder+'Best_solutions_metrics.txt','a') as best_evals:
			best_evals.write(f'Generation: {rep}\n')
			for i in range(eval_names.shape[0]):
				best_evals.write(str(eval_names[i])+':\t'+str(mean_std_list[best_indiv_pos][i][0])+u" \u00B1 "+str(mean_std_list[best_indiv_pos][i][1])+'\n')
			best_evals.write('Unweighted overall :\t'+str(evaluation_values[:-1,best_indiv_pos].mean())+'\n\n')

		#########

		with open(output_folder+'Overall_score_mean_std_per_generation.txt','a') as mean_std_file:
			for i,element in enumerate(mean_std_list[best_indiv_pos]):
				mean_std_file.write(f'{str(eval_names[i])}: {element[0]}'+u" \u00B1 "+f'{element[-1]}'+'\n')
			mean_std_file.write('\n')


		plt.title('Receiver Operating Characteristic')

		tprs = []
		aucs = []
		mean_fpr = np.linspace(0, 1, 100)

		fig, ax = plt.subplots()
		for i in range(len(roc_auc_list[best_indiv_pos])):
		    interp_tpr = np.interp(mean_fpr, roc_auc_list[best_indiv_pos][i][0], roc_auc_list[best_indiv_pos][i][1])
		    interp_tpr[0] = 0.0
		    tprs.append(interp_tpr)
		    aucs.append(roc_auc_list[best_indiv_pos][i][-1])
		    ax.plot(roc_auc_list[best_indiv_pos][i][0], roc_auc_list[best_indiv_pos][i][1], color='green', label = None, linewidth=0.4, alpha=0.1)

		ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="r", label="Random AUC = 0.5", alpha=0.8)

		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		std_auc = np.std(aucs)
		ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
		    lw=2, alpha=0.8)

		std_tpr = np.std(tprs, axis=0)
		tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
		ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

		ax.set(title=f"ROC/AUC MEvA-XGBoost P:{population} G:{generations}")
		ax.set_xlabel("False Positive Rate")
		ax.set_ylabel("True Positive Rate")

		ax.legend(loc="lower right", prop={'size': 6})
		plt.tight_layout()
		plt.savefig(output_folder+f'{rep}_ROC.png',dpi=600)
		plt.close()

		##############

		with open(output_folder+'Avg_solutions_metrics.txt','a') as avg_evals:
			avg_evals.write(f'Generation: {rep}\n')
			for i in range(eval_names.shape[0]):
				avg_evals.write(str(eval_names[i])+':\t'+str(evaluation_values[i].mean())+u" \u00B1 "+str(np.std(evaluation_values[i]))+'\n')
				#avg_evals.write(str(eval_names[i])+':\t'+str(evaluation_values[i].mean())+'\n')
			avg_evals.write('Unweighted overall :\t'+str(evaluation_values[:-1].mean())+'\n\n')


		average_performance = evaluation_values[:-1].mean()#np.average(evaluation_values[-1,:])
		best_performance = np.max(evaluation_values[:-1].mean(axis=0))#np.max(evaluation_values[-1])

		logging.info("Best Performance = " + str(best_performance))
		logging.info("Average Performance = " + str(average_performance))
		if average_performance != 0:
			logging.info("Convergence Percentage = " + str(math.fabs(evaluation_values[:-1,0].mean()-average_performance)/average_performance))
		else:
			logging.info("Convergence Percentage is impossible to calculate because average_performance = 0")
			print("Convergence Percentage is impossible to calculate because average_performance = 0")

		print(f'Average performance: {average_performance}')

		#Convergence criterion is checked in order to stop the evolution if the population is deemd us converged
		with open(output_folder+'Convergence.txt','a') as f:
			f.write(f'Generation {rep}\nConvergenve = {math.fabs(evaluation_values[:-1,0].mean()-average_performance)}\n')
			f.write(f'Convergence Percentage = {math.fabs(best_performance - average_performance)/best_performance}\n\n')
			#f.write(f'Convergence Percentage = {math.fabs(evaluation_values[:-1,0].mean()-average_performance)/best_performance}\n\n')
			#f.write(f'Convergence Percentage = {math.fabs(evaluation_values[:-1,0].sum()-average_performance)/average_performance}\n\n')

		if math.fabs(evaluation_values[-1,0]-average_performance)<0.001*average_performance: #Check if the 'Best' individual is almost equal to average
			#premature_termination=1
			print('Premature termination!')
			logging.info('Premature termination!')
			break


		Pareto_time_start = time.time()
		#Estimate non dominated fronts (Pareto fronts)
		fronts = pareto_frontiers(evaluation_values)

		with open(output_folder+'Pareto_fronts.txt','a') as pfronts:
			pfronts.write(f'Generation: {rep}\n')
			for i in np.sort(np.unique(fronts)):
				pfronts.write(f'Pareto {i}: {np.argwhere(fronts==i).squeeze()}\n')
			pfronts.write('\n')

		logging.info("Calculated Pareto Frontiers:")
		#print(fronts)
		logging.info(fronts)
		Pareto_time_stop = time.time()
		with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Generation: {rep}\nPareto frontier operation took: {Pareto_time_stop-Pareto_time_start}\n')

		pareto_distance_time_start = time.time()

		# Αpply selection operator: Ranked base selection is used

		# Tune fitness values by locating and using solution niches
		sigma_share = 0.5/(float(individuals.shape[1])**(0.1))
		#sigma_share = 0.1/(float(individuals.shape[1])**(0.2))

		# Use the rounded version aka XOR distance between individual solutions
		# evaluation_values_1 = similarity_function_rounded(fronts, evaluation_values, individuals, sigma_share, max_values, min_values)
		# Normal gene - gene distance between individual solutions
		evaluation_values_1 = similarity_function_rounded(fronts, evaluation_values, individuals, sigma_share, max_values, min_values)

		evaluation_values = evaluation_values_1.copy()
		logging.info("evaluation_values=")
		logging.info(evaluation_values)

		# Update the overall score based on the tuned fitness values
		# Eval_values_overall = [(b0*x0)+(b1*x1)+(b2*x2)+...+(b_n-2*x_n-2)+(b_n-1*x_n-1)]
		n_eval_values = evaluation_values.shape[0]-1
		for i in range(population): # For all individuals
			evaluation_values[-1,i] = (np.multiply(evaluation_values[:-1,i],goal_significances.T).sum())/n_eval_values

		pareto_distance_time_stop = time.time()
		with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Calculating the similarities of Niches / pareto frontiers (m) took: {pareto_distance_time_stop - pareto_distance_time_start}\n')
			time_file.write(f'# of fronts:{list(set(fronts))}\n')



		# NEW Generation preparation
		selected_individuals[0]=individuals[best_indiv_pos] # The individual with the highest overall_score before niches

		# Keep the parameter values and the Activated genes of the 'Best'/0-th individual in a file
		on_genes_indx = np.arange(individuals[best_indiv_pos,parameters:].shape[0])[individuals[best_indiv_pos,parameters:]>0.5]
		with open(output_folder+"best_solutions_active_genes.txt",'a') as best_solutions_fid:
			best_solutions_fid.write('Generation: '+str(rep)+'\n')
			best_solutions_fid.write(str(individuals[best_indiv_pos,np.append(np.arange(parameters),parameters+on_genes_indx)])+"\n")
			best_solutions_fid.write('Active Features: '+str(on_genes_indx)+'\n\n')


		# Keep the parameter values and all the non-zero genes of the 'Best'/0-th individual in a file
		on_genes_indx = np.arange(individuals[best_indiv_pos,parameters:].shape[0])[individuals[best_indiv_pos,parameters:]>0]
		with open(output_folder+"best_solutions_all_nonzero_genes.txt",'a') as best_solutions_all_fid:
			best_solutions_all_fid.write('Generation: '+str(rep)+'\n')
			best_solutions_all_fid.write(str(individuals[best_indiv_pos,np.append(np.arange(parameters),parameters+on_genes_indx)])+"\n")
			best_solutions_all_fid.write('All non-zero Features: '+str(on_genes_indx)+'\n\n')



		preparation_for_crossover_time_start = time.time()
		sum_ranked = evaluation_values[-1].sum() # Find the sum of overall scores for all individuals

		sum_ranked_eval_per_generation[rep] = sum_ranked

		sum_prop = np.zeros(population+1)
		sum_change = np.zeros(population)

		if sum_ranked > 0:
			for i in range(1, population+1):
				sum_prop[i] = sum_prop[i-1] + (evaluation_values[-1,i-1]/float(sum_ranked_eval_per_generation[rep])) ##Check this out
				sum_change[i-1] = (evaluation_values[-1,i-1]/float(sum_ranked_eval_per_generation[rep])) #17/11/2020


		with open(output_folder+'Roulette_change.txt','a') as roulette:
			roulette.write(f'Generation {rep}\n{sum_change}\n')


		# Roulette is based on the sum_change not the sum_prop. Essentially it is the probability of a solution to be selected
		# From 100 s=individuals pick 99 with Pr(i) = sum_change[i]. With replacement (can pick the same solution more than once)
		sel_indx = np.random.choice(a=population, size=population-1, replace=True, p=sum_change)
		for i in range(population-1):
			selected_individuals[i+1] = individuals[sel_indx[i]]

		# Save the selected indices in a file
		with open(output_folder+'selected_indiv_index.txt','a') as Sel_ind:
			Sel_ind.write(f'Generation {rep}\nIndividual indices selected: {str(sel_indx)}\n-------\n')



		preparation_for_crossover_time_stop = time.time()
		with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Preparation_for_crossover_time = {preparation_for_crossover_time_stop - preparation_for_crossover_time_start}\n')


		# Apply crossover operator
		cross_over_time_start = time.time()


		for i in range(1,population-1,2):
			random_number = np.random.uniform(size=population-1) #crete an array of random numbers that is used to activate the cross-over of a pair of individuals or not
			if random_number[i]<=two_points_crossover_probability:
				#print("Two Point Crossover")
				cross_point1=0
				cross_point2=0
				while cross_point1==cross_point2:
					cross_point1=math.ceil((individuals[0].shape[0]-parameters)*np.random.uniform(0,1))
					if cross_point1<math.floor((2*individuals[0].shape[0]-1)/3):
						width=math.ceil((math.floor(individuals[0].shape[0]-1)/3 -2)*np.random.uniform(0,1))
						cross_point2=cross_point1+width
					else:
						width=math.ceil(np.random.uniform(0,1)*(math.floor(individuals[0].shape[0]/3 -1)-2-(cross_point1-math.floor(2*individuals[0].shape[0]/3))))
						cross_point2=cross_point1+width
				if cross_point1>cross_point2:
					temp_cross_point=cross_point1
					cross_point1=cross_point2
					cross_point2=temp_cross_point

				cross_point1=int(cross_point1)
				cross_point2=int(cross_point2)

				with open(output_folder+'Mutation_points.txt', 'a') as tf:
					tf.write(f'Indiv_{i}, Indiv_{i+1} --> cross_point1: {cross_point1}, cross_point2: {cross_point2}, width: {width}'+'\n')


				# Create the children for the next generation
				temp_cross_over = selected_individuals[i+1, parameters+cross_point1:parameters+cross_point2].copy()
				selected_individuals[i+1, parameters+cross_point1:parameters+cross_point2] = selected_individuals[i, parameters+cross_point1:parameters+cross_point2].copy()
				selected_individuals[i, parameters+cross_point1:parameters+cross_point2] = temp_cross_over.copy()

			elif random_number[i]>two_points_crossover_probability and random_number[i]<(two_points_crossover_probability+arithmetic_crossover_probability):
				alpha=np.random.uniform(0,1)

				child1 = alpha*selected_individuals[i] + (1-alpha)*selected_individuals[i+1]
				child2 = (1-alpha)*selected_individuals[i] + alpha*selected_individuals[i+1]

				selected_individuals[i] = child1
				selected_individuals[i+1] = child2
				with open(output_folder+'Arithmetic_Mutations.txt', 'a') as tf:
					tf.write(f"Generation: {rep}:\n Indiv_{i}, Indiv_{i+1} --> alpha = {alpha} => CH1 = alpha*A + (1-alpha)*B , CH2 = (1-alpha)*A + alpha*B"+'\n')

		cross_over_time_stop = time.time()
		with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Cross over time = {cross_over_time_stop - cross_over_time_start}\n')


		# Αpply mutation operator
		mutation_time_start = time.time()
		with open(output_folder+'Mutation_points.txt', 'a') as tf:
			tf.write(f'Generation: {rep}'+'\n')

		sele_indivs = mutation(selected_individuals, population, min_values, max_values, mutation_probability, generations, rep, mu=0, s=0.1)

		mutation_time_stop = time.time()
		with open(output_folder+'timing.txt','a') as time_file:
			time_file.write(f'Mutation took = {mutation_time_stop - mutation_time_start}\n')


		# Update the population with the offspings
		individuals = sele_indivs.copy()

		# Inform the user for the new "Best" solution
		if best_indiv_pos==0:
			print('Individual_0 is the best solution again!')
		else:
			print(f'Individual_{best_indiv_pos} is the New best solution')

	with open (output_folder+"best_solution.txt","a") as best_solution_fid:
		for gene in range(individuals.shape[1]):
			if gene < len(individuals[0])-1:
				best_solution_fid.write(str(individuals[0,gene])+"\t")
			else:
				best_solution_fid.write(str(individuals[0,gene]))
		best_solution_fid.write("\n")

	return np.array(individuals)

def biomarker_discovery_modeller(dataset, feature_names, sample_names, labels, min_values, max_values, population,
								generations, two_points_crossover_probability=0.45, arithmetic_crossover_probability=0.45,
								mutation_probability=0.05, goal_significances=None, num_of_folds=10, output_folder=None,
								eval_names=None, missing_values_flag=True, normalize_flag=False):
	'''
	Main function of the MEvA-X tool that calls the other modules. Preprocessing steps and results are parts of this function
	Parameters:
	-----------
		dataset [ndarray]: a 2D array of the datapoints in the format of Rows:Features and Columns:Samples
		feature_names [ndarray]: a 1D array of the names of the features (genes) in the dataset
		sample_names [ndarray]: a 1D array of the names of the Samples in the dataset
		labels [ndarray]: a 1D array of the Labels for each Samples in the dataset
		min_values [ndarray]: a 1D array of the minimum allowed values for the parameters of the individuals
		max_values [ndarray]: a 1D array of the minimum allowed values for the parameters of the individuals
		population [int]: The number of individual solutions in the population
		generations [int]: The number of iterations to run the genetic algorithm for
		two_points_crossover_probability [float]: probability of the offsprings to be produced by the exchange of pieces from the parental individuals
		arithmetic_crossover_probability [float]: probability of an arithmetic crossover of the parental individuals to produce the offsprings
		mutation_probability [float]: probability of an offspring to mutate
		goal_significances [ndarray]: array of weights for the objectives
		num_of_folds [int]: Number of folds for cross validation
		output_folder [str]: path to the directory where to save the results
		eval_names [ndarray]: 1D array with the names of the evaluation metrics
		missing_values_flag [bool]: If True, try to impute missing values
		normalize_flag [bool]: If True, normalize the data
	Return:
	-----------
	'''
	print(f"Number of parameters = {parameters}")
	original_dataset = dataset.copy() # Keep a acopy of the data just in case!
	unique_labels = list(set(labels)) # or use np.unique(labels) <-- <numpy.ndarray>
	labels, unique_labels = transform_labels_to_numeric(labels,unique_labels)
	unique_labels = np.sort(unique_labels, axis=None)

	multiclass = True
	# If there is a two class problem, don't use Manhatan_dist
	if unique_labels.shape[0]==2:
		multiclass = False
		goal_significances = np.delete(goal_significances,-2)
		eval_names = np.delete(eval_names,-2)
		#goal_significances[-2]=0

	# Impute missing values
	if missing_values_flag:
		dataset_with_missing_values = impute(labels,dataset)
		dataset = dataset_with_missing_values.copy()

	# Normalize data in scale [0,1]
	if normalize_flag:
		dataset_normalized = normalize_dataset(dataset)
		dataset = dataset_normalized.copy()

	classification_problems = create_different_classification_problems(labels, unique_labels)

	min_values = np.append(min_values, np.zeros(dataset.shape[0]))
	max_values = np.append(max_values, np.ones(dataset.shape[0]))

	individuals = initialize_individuals(min_values, max_values, population)

	# Calculate or import the genes with FS methods
	if FS_calc:
		try:
			#FS_methods = mifs_calc(dataset, labels, k_vals=k_vals, n_features=n_features)
			JMI_genes = mifs_calc(dataset, labels, k_vals=k_vals, n_features=n_features, method='JMI')#FS_methods[:6]
			JMI_genes = JMI_genes.astype(np.int32, copy=False)
		except:
			print('JMI Feature selection failed. No filtering will apply on the features.')
			JMI_genes = None

		try:
			mRMR_genes = mifs_calc(dataset, labels, k_vals=k_vals, n_features=n_features, method='MRMR')#FS_methods[7:]
			mRMR_genes = mRMR_genes.astype(np.int32, copy=False)
		except:
			print('mRMR Feature selection failed. No filtering will apply on the features.')
			mRMR_genes = None

		try:
			Wilcoxon_genes = np.argwhere(Wilcoxon_ranksums(dataset, classification_problems)>0).squeeze()
			Wilcoxon_genes = Wilcoxon_genes.astype(np.int32, copy=False)
			print(Wilcoxon_genes.shape)
		except:
			print('Wilcoxon rank sums Feature selection failed. No filtering will apply on the features.')
			Wilcoxon_genes = None

		try:
			k_bst=100
			SelKBest_genes = SelectKBest(mutual_info_classif, k=k_bst).fit(dataset.T,labels) # This is an <object> you can use .scores_ to get the best features
			SelKBest_genes = np.flip(np.argsort(SelKBest_genes.scores_))[:k_bst]
			SelKBest_genes = SelKBest_genes.astype(np.int32, copy=False)
			print(SelKBest_genes.shape)
		except:
			print('Select K Best (SKB) Feature selection failed. No filtering will apply on the features.')
			SelKBest_genes = None
	else:
		# JMI
		try:
			try:
				JMI_genes = pd.read_csv(FS_dir+'JMI.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
			except:
				JMI_genes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Feature_selection_methods_final_13-11-2020/JMI.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
		except:
			print('No such a file found for JMI. No filtering will apply on the features')
			JMI_genes = None

		# mRMR
		try:
			try:
				mRMR_genes = pd.read_csv(FS_dir+'mRMR.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
			except:
				mRMR_genes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Feature_selection_methods_final_13-11-2020/mRMR.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
		except:
			print('No such a file found for mRMR. No filtering will apply on the features')
			mRMR_genes = None

		# Wilcoxon
		try:
			try:
				Wilcoxon_genes = pd.read_csv(FS_dir+'Wilcoxon_ranksums.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
			except:
				Wilcoxon_genes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Feature_selection_methods_final_13-11-2020/Wilcoxon_ranksums.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
		except:
			print('No such a file found for Wilcoxon rank sums. No filtering will apply on the features')
			Wilcoxon_genes = None

		# SKB
		try:
			try:
				SelKBest_genes = pd.read_csv(FS_dir+'Select_K_Best.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
			except:
				SelKBest_genes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Feature_selection_methods_final_13-11-2020/Select_K_Best.tsv', sep='\t', header = None).to_numpy().squeeze().astype(np.int32,copy=False)
		except:
			print('No such a file found for Select k best. No filtering will apply on the features')
			SelKBest_genes = None


	individuals = apply_evolutionary_process(generations, population, max_values, min_values, two_points_crossover_probability,
											arithmetic_crossover_probability, mutation_probability,dataset,
											labels, individuals, goal_significances, num_of_folds, classification_problems,
											output_folder, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes, eval_names, multiclass)


	evaluation_values,mean_std_list, roc_auc_list, N_trees = evaluate_individuals(dataset, labels, individuals, goal_significances, num_of_folds,
											classification_problems, output_folder, JMI_genes, Wilcoxon_genes,
											mRMR_genes, SelKBest_genes, multiclass)
	evaluation_values = np.array(evaluation_values, dtype = float)

	average_performance = np.mean(evaluation_values[-1])
	fronts = pareto_frontiers(evaluation_values)

	with open(output_folder+'Pareto_fronts.txt','a') as pfronts:
		pfronts.write('\nLast Pareto:\n')
		for i in np.sort(np.unique(fronts)):
			pfronts.write(f'Pareto {i}: {np.argwhere(fronts==i).squeeze()}\n')
		pfronts.write('\n')

	pareto1_indx = np.argwhere(fronts==1).squeeze()
	print(f'The indices of individuals in the 1st Pareto are: {pareto1_indx}')
	np.savetxt(output_folder+'Final_solutions.txt', delimiter=',', X=individuals[pareto1_indx],fmt='%.5f')

	filter_mask = filter_function(individuals, JMI_genes, Wilcoxon_genes, mRMR_genes, SelKBest_genes)
	feature_matrix = np.tile(feature_names,(individuals.shape[0],1))

	#################################################################################

	# Save the first pareto frontiers in a file based on the evaluation scores
	pareto1_maximums = np.argmax(evaluation_values[:,pareto1_indx],axis=1) # indices for the possition of the best score for each metric
	first_pareto_individuals = individuals[pareto1_indx].copy() #sub-matrix of the individuals in the first pareto front
	if not os.path.exists(output_folder+'Pareto_1_results/'):
		os.makedirs(output_folder+ 'Pareto_1_results/')
		os.makedirs(output_folder+ 'Pareto_1_results/Models/')

	for i,eval_name in enumerate(eval_names):
		np.savetxt(output_folder+f'Pareto_1_results/{eval_name}.txt', delimiter=',', X=first_pareto_individuals[pareto1_maximums[i],:].reshape(1,-1),fmt='%.5f')
		np.savetxt(output_folder+f'Pareto1_metrics_ALL_{eval_name}.csv', delimiter=',', X=evaluation_values[:,pareto1_indx],fmt='%.5f') #.reshape(1,-1)


	print('\nTraining of the final models started\n')
	# Train and save the final models

	################# MAJORITY VOTING START #######################
	
	print('Full Pareto majority voting')
	majority_voting(individuals, dataset, labels, filter_mask, pareto1_indx, num_of_folds, N_trees, multiclass, output_folder, par_type='full_pareto1')
	
	print('Highest Pareto values majority voting')
	majority_voting(individuals, dataset, labels, filter_mask, pareto1_indx[pareto1_maximums], num_of_folds, N_trees, multiclass, output_folder, par_type='top_pareto1')

	################# MAJORITY VOTING END #######################

	last_features = feature_names[individuals[0,parameters:]>0.5]
	print('Individual\'s 0 selected features:')
	print(np.asmatrix(last_features.squeeze()))
	np.savetxt(output_folder+'Potential_Biomarkers.txt', last_features, delimiter=',', fmt='%s')

	## All Pareto front 1 features Union
	Pareto1_features_union = feature_names[np.argwhere(np.sum(first_pareto_individuals[pareto1_maximums,parameters:]>0.5,axis=0)>0).squeeze()]
	print('Union of selected features for individuals in pareto front 1 with maximum evaluation values:')
	print(np.asmatrix(Pareto1_features_union.squeeze()))
	np.savetxt(output_folder+'Potential_Biomarkers_Union.txt', Pareto1_features_union, delimiter=',', fmt='%s')

	## All Pareto front 1 features Intersection
	Pareto1_features_intersect = feature_names[np.argwhere(np.sum(first_pareto_individuals[pareto1_maximums,parameters:]>0.5,axis=0)==first_pareto_individuals.shape[0]).squeeze()]
	print('Intersection of selected features for individuals in pareto front 1 with maximum evaluation values:')
	print(np.asmatrix(Pareto1_features_intersect.squeeze()))
	np.savetxt(output_folder+'Potential_Biomarkers_Intersection.txt', Pareto1_features_intersect, delimiter=',', fmt='%s')


	end_time = time.time()-prog_time
	msg = f'Program ended in : {end_time:.4} seconds or {end_time/60:.4} minutes\n'
	print(msg)
	with open(output_folder+'timing.txt','a') as time_file:
		time_file.write(msg)

	##################################### Saving the variables #####################################
	if not os.path.exists(os.path.join(output_folder,'Variables')):
		os.makedirs(os.path.join(output_folder,'Variables'))
	picklefy_variables(individuals,'individuals',output_folder)
	picklefy_variables(fronts,'fronts',output_folder)
	picklefy_variables(evaluation_values,'evaluation_values',output_folder)
	picklefy_variables(mean_std_list,'mean_std_list',output_folder)
	picklefy_variables(roc_auc_list,'roc_auc_list',output_folder)
	################################################################################################

def picklefy_variables(var,var_name,output_folder):
	try:
		pickle.dump(var, open(f'{output_folder}Variables/{var_name}.pkl', "wb"))
	except:
		print(f'An error occured while trying to save {var_name} variable')


def get_parser():
    '''
    This is a helper function to parse the inputs of the user from the command line into the variables used by the algorithm.

    '''
    # defined command line options
    # this also generates --help and error handling
    MEvAX_args = argparse.ArgumentParser(prog='MEvA-X', description='A hybrid algorithm for feature selection, hyper-parameter optimization and model training.\nMEvA-X uses a combination of a Nitched Pareto Evolutionary algorithm and the XGBoost Classifier to achieve the above-mentioned objectives.', epilog='This algorithm is the result of the work of K. Panagiotopoulos, K. Theofilatos, M.A. Deriu, and S.Mavroudi')

    MEvAX_args.add_argument("--dataset_path", "-A",  nargs=1, type=str,  default="data.csv", required=True, dest='dataset_filename', help="[str]: The path to the file containing the data. Format expected: FeaturesXSamples") # "*" -> 0 or more values expected => creates a list
    MEvAX_args.add_argument("--labels_path",, "-B"  nargs=1, type=str,  default="labels.csv", required=True, dest='labels_filename', help="[str]: The path to the file containing the labels of the data. Sample names should not be used")
    MEvAX_args.add_argument("--FS_dir",  nargs="*", type=str,  default=None, dest='FS_dir', help="[str]: The path to the directory containing precalculated features from the Feature Selection techniques (mRMR, JMI, Wilcoxon, and SelectKBest)")
    MEvAX_args.add_argument("--output_dir",  nargs="*", type=str,  default=os.path.join(os.getcwd(),"Results",f"Models_{str(time.time_ns())[:-9]}"), dest='output_folder')
    MEvAX_args.add_argument("--K",  nargs=1, type=int, default=10, dest='num_of_folds', help="[int]: The number of folds to be used in the K-fold cross validation. Default = 10")
    MEvAX_args.add_argument("--P",  nargs=1, type=int, default=50, dest='population', help="[int]: The number of individual solutions. Default = 50")
    MEvAX_args.add_argument("--G",  nargs=1, type=int, default=100, dest='generations', help="[int]: The number of maximum generations for the Evolutionry Algorithm. Default = 100")
    MEvAX_args.add_argument("--crossover_perc",  nargs=1, type=float, default=0.9, dest='two_points_crossover_probability')
    MEvAX_args.add_argument("--arithmetic_perc",  nargs=1, type=float, default=0.0, dest='arithmetic_crossover_probability', help="[float]: Probability of a two point crossover for the creation of a new offsping. Default = 100")
    MEvAX_args.add_argument("--mutation_perc",  nargs=1, type=float, default=0.05, dest='mutation_probability', help="[float]: The probability of point mutations to occure in the genome of an offspring. Default = 0.9")
    MEvAX_args.add_argument("--goal_sig_path",  nargs=1, type=str, dest='goal_significances_path', help="[str]: The path to the file containing the weights of the evaluation metrics")
    MEvAX_args.add_argument("--goal_sig_list",  nargs="*", type=list, default=[0.8,0.8,0.8,2,1,1,1,1,2,0.5,2], dest='goal_significances_filename', help="[array-like of floats]: The array of the weights for the evaluation metrics. Default = [0.8,0.8,0.8,2,1,1,1,1,2,0.5,2]")
    return MEvAX_args

''' = = = = = = = = = = = = = = = = = = = = = = = = MAIN = = = = = = = = = = = = = = = = = = = = = = = = '''


if __name__ == "__main__":
    prog_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    #-K 10 -P 50 -G 200 --dataset my_data.txt --labels my_labels.tsv -FS precalculated_features.csv --output_dir current_folder -cop 0.9 -acp 0 -mp 0.1 -goal_sig_lst 0.8 2 0.8 1 1 0.7 0.7 1 2 0.5 2
    dataset_filename = args.dataset_filename[0]
    labels_filename = args.labels_filename[0]
    FS_dir = args.FS_dir[0]
    output_folder = args.output_folder[0]
    num_of_folds = args.num_of_folds[0]
    population = args.population[0]
    generations = args.generations[0]
    two_points_crossover_probability = args.two_points_crossover_probability[0]
    arithmetic_crossover_probability = args.arithmetic_crossover_probability[0]
    mutation_probability = args.mutation_probability[0]
    goal_significances_filename = np.asarray(args.goal_significances_filename)

    # dataset_filename=sys.argv[1]
    # labels_filename=sys.argv[2]
    # population=int(sys.argv[3])
    # generations=int(sys.argv[4])
    # two_points_crossover_probability=float(sys.argv[5])
    # arithmetic_crossover_probability=float(sys.argv[6])
    # mutation_probability=float(sys.argv[7])	
    # goal_significances_filename=sys.argv[8]
    # num_of_folds=int(sys.argv[9])
    

    #####
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    #####
    current_dir = os.getcwd()
    FS_dir = os.path.join(current_dir, 'diet/FS_methods/')

    [dataset, feature_names, sample_names, labels] = preprocessing_function(dataset_filename,labels_filename, as_pandas=True)

    ####### PARAMETERS #######
    #						1	2	3	4		5	6	7	8	9	10	11	12 13
    min_values = np.array([0,	0,	4,	1,	0.01,	1,	0,	0,	0,	0,	0])#, 0.3, 0.3]) # 1.FS_method  2.use_of_FS  3.k-NN(mifs)  4.k_SKB  5.eta  6.max_depth  7.gamma
    max_values = np.array([5,	3,	11,	101, 0.35,	7,	10,	10,	8,	15,	5])#, 1.0, 1.0]) # 8.lambda  9.alpha  10.min_child_weight  11.scale_pos_weight  12.colsample  13.subsample
    parameters = max_values.shape[0]

    num_of_folds = int(input("K-fold = "))

    '''This section is here to test the Feature selection methods'''
    FS_calc = input("Do you need to calculate the feature selection algorithms? (Y/N): ")
    if FS_calc.upper() in ['Y','YES']:
        FS_calc = True
        try:
            k_vals = []
            print('Give a list of k (one at a time) for the KNN of the mutual information methods [4-10 is recommended]: ')
            while True:
                k_vals.append(int(input()))
                k_vals = list(np.unique(k_vals))
        except:
            pass
        n_features = int(input('Give the number of best features to keep: '))
    else:
        FS_calc = False
    '''up to this line.'''

    #two_points_crossover_probability = 0.9
    #arithmetic_crossover_probability = 0
    #mutation_probability = 0.05
    # G1:Feature_compl, G2:Acc, G3:Split_compl, G4:wGM, G5:F1, G6:F2, G7:Precision, G8:Recall, G9:AUC, G10:Manhatan_dist, G11:GM
    ans = str(input('Use the unweighted (default) metrics? (Y/N): '))
    if ans.upper() in ['Y','YES']:
        goal_significances = np.ones(11)
    else:
        goal_significances = np.array([0.8,0.8,0.8,2,1,1,1,1,2,0.5,2])

    print(f'Weights: {goal_significances}')

    tstamp = time.strftime('%Y_%m_%d')
    output_folder = os.path.join(dname,'XGB_Results_diet\\'+f'P{population}_G{generations}_K{num_of_folds}'+'_'+str(time.time())+'\\')
    # 	if dataset_filename == 'C:/Users/kosta/Desktop/Thesis/My_code/diet/diet_dataset.txt':
    # 		output_folder = 'C:/Users/kosta/Desktop/Thesis/My_code/Results_directory/'+str(tstamp)+'_'+str(time.time())+'/'
    # 	elif dataset_filename == os.path.join(os.path.expanduser('~'), 'My_code', 'diet', 'diet_dataset.txt'):
    # 		output_folder = os.path.join(os.path.expanduser('~'), 'My_code', 'Results_directory',f'{str(tstamp)}_{str(time.time())}/')
    # 	else:
    # 		output_folder = '/content/drive/MyDrive/Colab Notebooks/Results_directory/'+str(tstamp)+'_'+str(time.time())+'/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Select if normalization and imputation will take place
    missing_values_flag = True
    normalize_flag = True

    with open(output_folder+'Inputs.txt','a') as param_file:
        param_file.write(f'cwd: {current_dir}\ndataset file: {dataset_filename}\n')
        param_file.write(f'labels file: {labels_filename}\nNumber of parameters: {parameters}\n')
        param_file.write(f'Population: {population}\nGenerations: {generations}\n')
        param_file.write(f'k-folds: {num_of_folds}')

    eval_names = np.array(['Model_complexity #features','Accuracy','Model_complexity #splits',
                                'weighted Geometric Mean','F1 score','F2 score','Precision','Recall','AUrocC',
                                'Balanced_accuracy','Manhattan distance^-1','Overall_score'])

    biomarker_discovery_modeller(dataset, feature_names, sample_names, labels, min_values, max_values, population,
                                generations, two_points_crossover_probability, arithmetic_crossover_probability,
                                mutation_probability, goal_significances, num_of_folds, output_folder,
                                eval_names, missing_values_flag, normalize_flag)
