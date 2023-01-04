from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import math
from skorch import NeuralNetClassifier, NeuralNet
import skorch
from torch import nn
import torch
import numpy as np
import sys
sys.path.insert(1, '../')
import mlp_classifier

def train_rew(train_dataset, base_classifier, dataset):
    if dataset == 'adult':
        sensitive='sex'
        label='income'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'bank':
        sensitive = 'age'
        label='deposit'
        privileged_groups = [{'age':1}]
        unprivileged_groups = [{'age': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'credit':
        sensitive = 'sex'
        label='credit'
        privileged_groups = [{'sex':1}]
        unprivileged_groups = [{'sex': 0}]
        favorable_label = 1
        unfavorable_label = 0
    elif dataset == 'compas':
        sensitive = 'race'
        label = 'category'
        privileged_groups = [{'race':0}]
        unprivileged_groups = [{'race': 1}]
        favorable_label = 0
        unfavorable_label = 1
    elif dataset == 'synthetic':
        sensitive = 'sensitive'
        label='label'
        privileged_groups = [{'sensitive':1}]
        unprivileged_groups = [{'sensitive': 0}]
        favorable_label = 1
        unfavorable_label = 0
    preProc = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    preProc.fit(train_dataset)
    new_train_dataset = preProc.transform(train_dataset)
    if base_classifier == 'lr':
        model = LogisticRegression()
        model.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels.ravel(), sample_weight=new_train_dataset.instance_weights)
    elif base_classifier == 'svm':
        model = CalibratedClassifierCV()
        model.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels.ravel(), sample_weight=new_train_dataset.instance_weights)
    elif base_classifier == 'rf':
        model = RandomForestClassifier()
        model.fit(new_train_dataset.features[:,:-1], new_train_dataset.labels.ravel(), sample_weight=new_train_dataset.instance_weights)
    elif base_classifier == 'mlp':
        y_train = new_train_dataset.labels.ravel().astype(int)
        x_train = {'data': new_train_dataset.features[:,:-1].astype(np.float32),
            'sample_weight': new_train_dataset.instance_weights.astype(np.float32)}
        model = mlp_classifier.mlp_model(new_train_dataset.features.shape).fit(x_train, y_train)
    return model