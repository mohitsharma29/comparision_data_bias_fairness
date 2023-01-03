from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.inprocessing.grid_search_reduction import GridSearchReduction

def train_grid_search(train_dataset, base_classifier='lr', constraint='dp'):
    if constraint == 'dp':
        constraint = 'DemographicParity'
    elif constraint == 'eodds':
        constraint = 'EqualizedOdds'
    
    # Prot_attribute already dropped during model training, no need to drop again
    if base_classifier == 'lr':
        inProc = GridSearchReduction(LogisticRegression(), constraints=constraint, drop_prot_attr=False)
    elif base_classifier == 'svm':
        inProc = GridSearchReduction(SVC(probability=True), constraints=constraint, drop_prot_attr=False)
    elif base_classifier == 'rf':
        inProc = GridSearchReduction(RandomForestClassifier(), constraints=constraint, drop_prot_attr=False)
    
    inProc.fit(train_dataset)
    return inProc