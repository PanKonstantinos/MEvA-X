import sys
import os
import numpy as np
import pandas as pd
import argparse
import time

def sanity_check(args):
    assert type(args.dataset_filename) == str, "dataset_filename should be a string type object <str>"
    assert type(args.labels_filename) == str, "labels_filename should be a string type object <str>"
    assert type(args.FS_dir) == str, "FS_dir should be a string type object <str>"
    assert type(args.output_folder) == "output_folder", "x should be 'hello'"
    assert type(args.num_of_folds) == "num_of_folds", "x should be 'hello'"
    assert type(args.population) == "goodbye", "population should be 'hello'"
    assert type(args.generations) == "goodbye", "generations should be 'hello'"
    assert type(args.two_points_crossover_probability) in [float, int], "two_points_crossover_probability should be either a <float> or an <int>"
    assert type(args.arithmetic_crossover_probability) in [float, int], "arithmetic_crossover_probability should be either a <float> or an <int>"
    assert type(args.mutation_probability) in [float, int], "mutation_probability should be either a <float> or an <int>"
    assert type(args.goal_sig_path) in [float, int], "x should be 'hello'"
    assert type(args.goal_sig_list) in [np.ndarray, list], "goal_sig_list should be an array-like object"


def get_parser():
    '''
    This is a helper function to parse the inputs of the user from the command line into the variables used by the algorithm.

    '''
    # defined command line options
    # this also generates --help and error handling
    MEvAX_args = argparse.ArgumentParser(prog='MEvA-X', description='A hybrid algorithm for feature selection, hyper-parameter optimization and model training.\nMEvA-X uses a combination of a Nitched Pareto Evolutionary algorithm and the XGBoost Classifier to achieve the above-mentioned objectives.', epilog='This algorithm is the result of the work of K. Panagiotopoulos, K. Theofilatos, M.A. Deriu, and S.Mavroudi')

    MEvAX_args.add_argument("--dataset_path", "-A", type=str,  default="data.csv", required=True, dest='dataset_filename', help="[str]: The path to the file containing the data. Format expected: FeaturesXSamples") # "*" -> 0 or more values expected => creates a list
    MEvAX_args.add_argument("--labels_path", "-B", type=str,  default="labels.csv", required=True, dest='labels_filename', help="[str]: The path to the file containing the labels of the data. Sample names should not be used")
    MEvAX_args.add_argument("--FS_dir", type=str,  default=None, dest='FS_dir', help="[str]: The path to the directory containing precalculated features from the Feature Selection techniques (mRMR, JMI, Wilcoxon, and SelectKBest)")
    MEvAX_args.add_argument("--output_dir", type=str,  default=os.path.join(os.getcwd(),"Results",f"Models_{str(time.time_ns())[:-9]}"), dest='output_folder')
    MEvAX_args.add_argument("--K", type=int, default=10, dest='num_of_folds', help="[int]: The number of folds to be used in the K-fold cross validation. Default = 10")
    MEvAX_args.add_argument("--P", type=int, default=50, dest='population', help="[int]: The number of individual solutions. Default = 50")
    MEvAX_args.add_argument("--G", type=int, default=100, dest='generations', help="[int]: The number of maximum generations for the Evolutionry Algorithm. Default = 100")
    MEvAX_args.add_argument("--crossover_perc", type=float, default=0.9, dest='two_points_crossover_probability')
    MEvAX_args.add_argument("--arithmetic_perc", type=float, default=0.0, dest='arithmetic_crossover_probability', help="[float]: Probability of a two point crossover for the creation of a new offsping. Default = 100")
    MEvAX_args.add_argument("--mutation_perc", type=float, default=0.05, dest='mutation_probability', help="[float]: The probability of point mutations to occure in the genome of an offspring. Default = 0.9")
    MEvAX_args.add_argument("--goal_sig_path", type=str, dest='goal_significances_path', help="[str]: The path to the file containing the weights of the evaluation metrics")
    MEvAX_args.add_argument("--goal_sig_list",  nargs="*", type=list, default=[0.8,0.8,0.8,2,1,1,1,1,2,0.5,2], dest='goal_significances_filename', help="[array-like of floats]: The array of the weights for the evaluation metrics. Default = [0.8,0.8,0.8,2,1,1,1,1,2,0.5,2]")
    return MEvAX_args


if __name__ == "__main__":
    print("Welcome!")
    parser = get_parser()
    args = parser.parse_args()
    sanity_check(args)
    print(args)
    
    dataset_filename = args.dataset_filename
    labels_filename = args.labels_filename
    FS_dir = args.FS_dir
    output_folder = args.output_folder
    num_of_folds = args.num_of_folds
    population = args.population
    generations = args.generations
    two_points_crossover_probability = args.two_points_crossover_probability
    arithmetic_crossover_probability = args.arithmetic_crossover_probability
    mutation_probability = args.mutation_probability
    
    
    #for (var in vars(args)):
     #   print(var, vars(args, var))