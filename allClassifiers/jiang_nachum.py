import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import copy
import skorch
from torch import nn
import torch
import numpy as np
import sys
sys.path.insert(1, '../')
import mlp_classifier

# Identifying and correcting label bias in Machine Learning: https://arxiv.org/pdf/1901.04966.pdf#page=9&zoom=100,0,0
# Code taken from: https://github.com/google-research/google-research

def get_error_and_violations_dp(y_pred, y, protected_attributes):
    acc = np.mean(y_pred != y)
    violations = []
    for p in protected_attributes:
        protected_idxs = np.where(p > 0)
        violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    pairwise_violations = []
    for i in range(len(protected_attributes)):
        for j in range(i+1, len(protected_attributes)):
            protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
            if len(protected_idxs[0]) == 0:
                continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    return acc, violations, pairwise_violations

def debias_weights_dp(original_labels, protected_attributes, multipliers):
    exponents = np.zeros(len(original_labels))
    for i, m in enumerate(multipliers):
        exponents -= m * protected_attributes[i]
    weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
    weights = np.where(original_labels > 0, 1 - weights, weights)
    return weights

def get_error_and_violations_eodds(y_pred, y, protected_attributes):
    acc = np.mean(y_pred != y)
    violations = []
    for p in protected_attributes:
        protected_idxs = np.where(np.logical_and(p > 0, y > 0))
        positive_idxs = np.where(y > 0)
        violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
        protected_idxs = np.where(np.logical_and(p > 0, y < 1))
        negative_idxs = np.where(y < 1)
        violations.append(np.mean(y_pred[negative_idxs]) - np.mean(y_pred[protected_idxs]))
    pairwise_violations = []
    for i in range(len(protected_attributes)):
        for j in range(i+1, len(protected_attributes)):
            protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
            if len(protected_idxs[0]) == 0:
                continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    return acc, violations, pairwise_violations

def debias_weights_eodds(original_labels, predicted, protected_attributes, multipliers):
    exponents_pos = np.zeros(len(original_labels))
    exponents_neg = np.zeros(len(original_labels))

    for i, protected in enumerate(protected_attributes):
        exponents_pos -= multipliers[2 * i] * protected
        exponents_neg -= multipliers[2 * i + 1] * protected
    weights_pos = np.exp(exponents_pos)/ (np.exp(exponents_pos) + np.exp(-exponents_pos))
    weights_neg = np.exp(exponents_neg)/ (np.exp(exponents_neg) + np.exp(-exponents_neg))

    #weights = np.where(predicted > 0, weights, 1 - weights)
    weights = np.where(original_labels > 0, 1 - weights_pos, weights_neg)
    return weights

def get_error_and_violations_eop(y_pred, y, protected_attributes):
    acc = np.mean(y_pred != y)
    violations = []
    for p in protected_attributes:
        protected_idxs = np.where(np.logical_and(p > 0, y > 0))
        positive_idxs = np.where(y > 0)
        violations.append(np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs]))
    pairwise_violations = []
    for i in range(len(protected_attributes)):
        for j in range(i+1, len(protected_attributes)):
            protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
            if len(protected_idxs[0]) == 0:
                continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    return acc, violations, pairwise_violations

def debias_weights_eop(original_labels, predicted, protected_attributes, multipliers):
    exponents = np.zeros(len(original_labels))
    for i, m in enumerate(multipliers):
        exponents -= m * protected_attributes[i]
    weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
    #weights = np.where(predicted > 0, weights, 1 - weights)
    weights = np.where(original_labels > 0, 1 - weights, weights)
    return weights

def train_jiang_nachum_reweighing(train_dataset, modelType='lr', constraint='dp'):
    # Jiang, H., & Nachum, O. (2019). Identifying and correcting label bias in machine learning (https://github.com/google-research/google-research/tree/master/label_bias)
    # Supports any base classifier with instance weights
    X_train = train_dataset.features[:,:-1]
    #X_test = test_dataset.features[:, :-1]
    y_train = train_dataset.labels.squeeze()
    #y_test = test_dataset.labels.squeeze()
    protected_train = train_dataset.protected_attributes.squeeze()
    protected_train_pos = copy.deepcopy(protected_train)
    protected_train_neg = 1 - copy.deepcopy(protected_train)
    protected_train = [protected_train_pos, protected_train_neg]
    #protected_test = [test_dataset.protected_attributes.squeeze()]
    multipliers = np.zeros(len(protected_train)*2)
    weights = np.array([1] * X_train.shape[0])
    learning_rate = 1.
    n_iters = 100
    for it in range(n_iters):
        if modelType == 'lr':
            model = LogisticRegression()
            model.fit(X_train, y_train, weights)
            y_pred_train = model.predict(X_train)
        elif modelType == 'svm':
            model = CalibratedClassifierCV()
            model.fit(X_train, y_train, weights)
            y_pred_train = model.predict(X_train)
        elif modelType == 'rf':
            model = RandomForestClassifier()
            model.fit(X_train, y_train, weights)
            y_pred_train = model.predict(X_train)
        elif modelType == 'mlp':
            train_set = {'data': X_train.astype(np.float32),
            'sample_weight': weights.astype(np.float32)}
            model = mlp_classifier.mlp_model(train_dataset.features.shape).fit(train_set, y_train.astype(int))
            pred_train_set = {'data': X_train.astype(np.float32),
            'sample_weight': np.ones_like(y_train).astype(np.float32)}
            y_pred_train = np.argmax(model.predict(pred_train_set), axis=1)
        
        if constraint == 'dp':
            weights = debias_weights_dp(y_train, protected_train, multipliers)
            acc, violations, pairwise_violations = get_error_and_violations_dp(y_pred_train, y_train, protected_train)
        elif constraint == 'eodds':
            weights = debias_weights_eodds(y_train, y_pred_train, protected_train, multipliers)
            acc, violations, pairwise_violations = get_error_and_violations_eodds(y_pred_train, y_train, protected_train)
        elif constraint == 'eop':
            weights = debias_weights_eop(y_train, y_pred_train, protected_train, multipliers)
            acc, violations, pairwise_violations = get_error_and_violations_eop(y_pred_train, y_train, protected_train)
        multipliers += learning_rate * np.array(violations)
    return model