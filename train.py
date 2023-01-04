import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
from tqdm import tqdm as tqdm

from allClassifiers import reweighing, exp_grad, aif_post_proc, jiang_nachum, prej_remover, gerry_fair, grid_search

import mlp_classifier

from data_utils import preprocessDataset
from jn_preprocessing import preprocessAdult
from utils import test_preproc, test_inproc

import pickle
import math
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--base_classifier", type=str,
                    help="Base Classifier to use", default='lr')
parser.add_argument("--dataset", type=str,
                    help="Dataset to run on", default='adult')
parser.add_argument("--algorithm", type=str,
                    help="Algorithm to use", default='base')
parser.add_argument("--constraint", type=str, 
                    help='Fairness Constraint (used only when required)', default='eop')
parser.add_argument("--run_index", type=str, 
                    help='Run index', default='1')
args = parser.parse_args()

if __name__ == '__main__':
    # Underrepresentation Beta_Pos and Beta_Neg results
    # First index of the dictionary will always have the AIF metric object, further indices might have algorithm specific metrics
    results = {}
    results['undersample'] = {}
    print('Performing Under-Representation Experiments')
    for i in tqdm(os.listdir(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/betaDatasets/')):
        try:
            train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)
            """if args.dataset == 'adult':
                #train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)
                train_dataset, test_dataset_original = preprocessAdult(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv')
            elif args.dataset == 'credit':
                train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)"""
            #_, test_dataset_balanced = preprocessDataset(f'/media/data_dump/Mohit/neurips2022_data/{args.dataset}/raw/betaDatasets/{i}', f'/media/data_dump/Mohit/neurips2022_data/{args.dataset}/balanced/balanced_test.csv', args.dataset)
            test_datasets = {
                #'balanced':test_dataset_balanced,
                #'biased':test_dataset_biased,
                'original':test_dataset_original
            }
            
            if args.algorithm == 'base':
                if args.base_classifier == 'lr':
                    model = LogisticRegression().fit(train_dataset.features[:,:-1], train_dataset.labels)
                elif args.base_classifier == 'rf':
                    model = RandomForestClassifier().fit(train_dataset.features[:,:-1], train_dataset.labels)
                elif args.base_classifier == 'svm':
                    model = CalibratedClassifierCV().fit(train_dataset.features[:,:-1], train_dataset.labels)
                elif args.base_classifier == 'mlp':
                    y_train = train_dataset.labels.ravel().astype(int)
                    x_train = {'data': train_dataset.features[:,:-1].astype(np.float32),
                        'sample_weight': np.ones_like(y_train).astype(np.float32)}
                    model = mlp_classifier.mlp_model(train_dataset.features.shape).fit(x_train, y_train)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset, args.base_classifier)
            elif args.algorithm == 'rew':
                model = reweighing.train_rew(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset, args.base_classifier)
            elif args.algorithm == 'jiang_nachum':
                model = jiang_nachum.train_jiang_nachum_reweighing(train_dataset, modelType=args.base_classifier, constraint=args.constraint)
                results['undersample'][i] = test_preproc(model, test_datasets, args.dataset)
            elif args.algorithm == 'exp_grad':
                model = exp_grad.train_exp_grad(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
                results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            elif args.algorithm == 'grid_search':
                model = grid_search.train_grid_search(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
                results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            elif args.algorithm == 'prej_remover':
                model = prej_remover.train_prej_remover(train_dataset, dataset=args.dataset)
                results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            elif args.algorithm == 'gerry_fair':
                model = gerry_fair.train_gerry_fair(train_dataset)
                results['undersample'][i] = test_inproc(model, test_datasets, args.dataset)
            elif args.algorithm == 'eq':
                results['undersample'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='eq', base_classifier=args.base_classifier, dataset=args.dataset)
            elif args.algorithm == 'cal_eq':
                results['undersample'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='cal_eq', base_classifier=args.base_classifier, dataset=args.dataset)
            elif args.algorithm == 'reject':
                results['undersample'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='reject', base_classifier=args.base_classifier, dataset=args.dataset)
        except Exception as e:
            print(i, args.algorithm)
            print(e)
            raise
    
    # Label Bias Results
    results['label_bias'] = {}
    print()
    print('Performing Label Bias Experiments')
    for i in tqdm(os.listdir(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/labelBiasDatasets/')):
        train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/labelBiasDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)
        """if args.dataset == 'adult':
            #train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/labelBiasDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)
            train_dataset, test_dataset_original = preprocessAdult(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/labelBiasDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv')
        elif args.dataset == 'credit':
            train_dataset, test_dataset_original = preprocessDataset(f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/labelBiasDatasets/{i}', f'/media/data_dump/Mohit/facct23_samplebias_data/run_{args.run_index}/{args.dataset}/raw/original_test.csv', args.dataset)"""
        test_datasets = {
            #'balanced':test_dataset_balanced,
            #'biased':test_dataset_biased,
            'original':test_dataset_original
        }
        if args.algorithm == 'base':
            if args.base_classifier == 'lr':
                model = LogisticRegression().fit(train_dataset.features[:,:-1], train_dataset.labels)
            elif args.base_classifier == 'rf':
                model = RandomForestClassifier().fit(train_dataset.features[:,:-1], train_dataset.labels)
            elif args.base_classifier == 'svm':
                model = CalibratedClassifierCV().fit(train_dataset.features[:,:-1], train_dataset.labels)
            elif args.base_classifier == 'mlp':
                y_train = train_dataset.labels.ravel().astype(int)
                x_train = {'data': train_dataset.features[:,:-1].astype(np.float32),
                    'sample_weight': np.ones_like(y_train).astype(np.float32)}
                model = mlp_classifier.mlp_model(train_dataset.features.shape).fit(x_train, y_train)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset, args.base_classifier)
        elif args.algorithm == 'rew':
            model = reweighing.train_rew(train_dataset, base_classifier=args.base_classifier, dataset=args.dataset)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset, args.base_classifier)
        elif args.algorithm == 'jiang_nachum':
            model = jiang_nachum.train_jiang_nachum_reweighing(train_dataset, modelType=args.base_classifier, constraint=args.constraint)
            results['label_bias'][i] = test_preproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'exp_grad':
            model = exp_grad.train_exp_grad(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'grid_search':
            model = grid_search.train_grid_search(train_dataset, base_classifier=args.base_classifier, constraint=args.constraint)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'prej_remover':
            model = prej_remover.train_prej_remover(train_dataset, dataset=args.dataset)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'gerry_fair':
            model = gerry_fair.train_gerry_fair(train_dataset)
            results['label_bias'][i] = test_inproc(model, test_datasets, args.dataset)
        elif args.algorithm == 'eq':
            results['label_bias'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='eq', base_classifier=args.base_classifier, dataset=args.dataset)
        elif args.algorithm == 'cal_eq':
            results['label_bias'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='cal_eq', base_classifier=args.base_classifier, dataset=args.dataset)
        elif args.algorithm == 'reject':
            results['label_bias'][i] = aif_post_proc.train_aif_post_proc(train_dataset, test_datasets, algorithm='reject', base_classifier=args.base_classifier, dataset=args.dataset)
    # Store Results
    with open(f'/media/data_dump/Mohit/facct23_samplebias_data/results/run_{args.run_index}/{args.algorithm}__{args.constraint}__{args.dataset}__{args.base_classifier}.pkl', 'wb') as f:
        pickle.dump(results, f)