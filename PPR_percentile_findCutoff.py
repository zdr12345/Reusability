import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def PPR_percentile_findCutoff(X, Y, clf, precisionDesired, k_fold, isSMOTE):
    '''
    Using k-fold cv to find the best cutoff. In each fold, we find the corresponding precision and recall of e
    ach percentile from 50-99. Then calculate the average PPHx across k-fold.
    :param X: predictors
    :param Y: response
    :param clf: classifier
    :param precisionDesired: the desired precision
    :param k_fold: number of folds
    :param isSMOTE: whether to oversampling
    :return: The percentile which has the highest average PPHx
    '''

    Precision = np.zeros([k_fold, 50])
    Recall = np.zeros([k_fold, 50])
    kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=2018)
    val = 0
    for train_idx, valid_idx in kf.split(X, Y):
        train_X = X[train_idx, :]
        train_Y = Y[train_idx]
        valid_X = X[valid_idx, :]
        valid_Y = Y[valid_idx]

        if isSMOTE == 1:
            smote = SMOTE(random_state=2018)
            smox, smoy = smote.fit_sample(train_X, train_Y)
        else:
            smox, smoy = train_X, train_Y

        clf = clf.fit(smox, smoy)
        Prob_valid = clf.predict_proba(valid_X)[:, 1]
        for idx, pct in enumerate(range(50, 100)):
            Precision[val, idx] = precision_score(valid_Y, Prob_valid >= np.percentile(Prob_valid, pct))
            if Precision[val, idx] < precisionDesired:
                Recall[val, idx] = 0
            else:
                Recall[val, idx] = recall_score(valid_Y, Prob_valid >= np.percentile(Prob_valid, pct))
        val += 1
    isDesiredPrecision = np.mean(1*(Precision>=precisionDesired), axis=0)
    Precision = np.mean(Precision, 0)
    Recall = np.mean(Recall, 0)
    Cutoff = np.argmax(isDesiredPrecision) + 50

    return([Cutoff, Precision[np.argmax(isDesiredPrecision)], Recall[np.argmax(isDesiredPrecision)]])


