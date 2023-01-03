from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification, EqOddsPostprocessing
import numpy as np
from aif360.metrics import ClassificationMetric

# https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_calibrated_eqodds_postprocessing.ipynb
def train_aif_post_proc(train_dataset, test_datasets, algorithm, base_classifier='lr', dataset='adult'):
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
    new_train_set, new_val_set = train_dataset.split([0.8], shuffle=True)
    if base_classifier == 'lr':
        lr_model = LogisticRegression().fit(new_train_set.features[:,:-1], new_train_set.labels.ravel())
    elif base_classifier == 'rf':
        lr_model = RandomForestClassifier().fit(new_train_set.features[:,:-1], new_train_set.labels.ravel())
    fav_idx = np.where(lr_model.classes_ == new_train_set.favorable_label)[0][0]
    
    class_thresh = 0.5
    pred_val_set = new_val_set.copy(deepcopy=True)
    y_valid_pred_prob = lr_model.predict_proba(pred_val_set.features[:,:-1])[:,fav_idx].reshape(-1,1)
    pred_val_set.scores = y_valid_pred_prob
    
    y_valid_pred = np.zeros_like(new_val_set.labels)
    y_valid_pred[y_valid_pred_prob >= class_thresh] = new_val_set.favorable_label
    y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = new_val_set.unfavorable_label
    pred_val_set.labels = y_valid_pred

    pred_test_sets = {}
    for i in test_datasets:
        pred_test_set = test_datasets[i].copy(deepcopy=True)
        y_test_pred_prob = lr_model.predict_proba(pred_test_set.features[:,:-1])[:,fav_idx].reshape(-1,1)
        pred_test_set.scores = y_test_pred_prob
        
        y_test_pred = np.zeros_like(test_datasets[i].labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = test_datasets[i].favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = test_datasets[i].unfavorable_label
        pred_test_set.labels = y_test_pred
        pred_test_sets[i] = pred_test_set
    
    if algorithm == 'eq':
        postProc = EqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    elif algorithm == 'cal_eq':
        postProc = CalibratedEqOddsPostprocessing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, cost_constraint="fnr")
    elif algorithm == 'reject':
        postProc = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups, 
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                  num_class_thresh=100, num_ROC_margin=50,
                                  metric_name="Average odds difference",
                                  metric_ub=0.05, metric_lb=-0.05)
    postProc.fit(new_val_set, pred_val_set)
    final_results = {}
    for i in pred_test_sets:
        final_results[i] = ClassificationMetric(test_datasets[i], postProc.predict(pred_test_sets[i]), unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    return final_results