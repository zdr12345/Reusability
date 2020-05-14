import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE, RFECV
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint

from PPR_percentile_score import PPR_percentile_score
from Filter_Ttest import Filter_Ttest
from PPR_percentile_findCutoff import PPR_percentile_findCutoff

def RepeatCV(times, Clinical, NumOfFea, Criteria, Method, NumFold = 10):
    '''
    k-fold CV. In each fold, the training set are further split into training and validation set.
    The training set is used to build the model.
    The validation set is used to determine the CUTOFF in order to ensure the precision.
    Once the cutoff is determined, it will be used on testing set to obtain precision and PPR.
    :param times: Number of repetitions
    
    :param Clinical: Which variable to predict. Should be index of columns.
    
    :param NumOfFea: Number of genes to be selected
    :param Criteria: the score used in building model, Possible options: FSCORE, PPR90, PPR95
    :param Method: RN_only, RN_SMOTE, no_RN_no_SMOTE
    :param NumFold: number of folds in k-fold CV
    :return: F05score_fold, PPR_fold, PRECISION_fold, F05PPR_fold, AUROC_fold, valid_Precision
    '''

    workDir = "/gpfs/home/dz16e/Reusability/NewExperiment/"
    if Criteria == 'FSCORE':
        myscorer = make_scorer(fbeta_score, beta=1.0)
        target_Precision = 0.9
    elif Criteria == 'FSCORE_95':
        myscorer = make_scorer(fbeta_score, beta=1.0)
        target_Precision = 0.95
    elif Criteria == 'PPR90':
        myscorer = make_scorer(PPR_percentile_score, needs_proba=True, Precision=0.9, Return_cutoff=0)
        target_Precision = 0.9
    elif Criteria == 'PPR95':
        myscorer = make_scorer(PPR_percentile_score, needs_proba=True, Precision=0.95, Return_cutoff=0)
        target_Precision = 0.95

    if Method in ['RN_only', 'RN_SMOTE']:
        X_data = np.loadtxt(workDir + 'TCGA_Data/Predictors/' + 'predictor_rank.txt')
    elif Method in ['no_RN_no_SMOTE']:
        X_data = np.loadtxt(workDir + 'TCGA_Data/Predictors/' + 'predictor.txt')
    Y = np.loadtxt(workDir + 'TCGA_Data/Responses/' + 'response.txt')

    CLI = Clinical
    Y_data = Y[:, CLI]
    X_data = X_data[Y_data < 2, :]
    Y_data = Y_data[Y_data < 2]
    
    F05score_fold = np.zeros([NumFold, 4])
    PPR_fold = np.zeros([NumFold, 4])
    AUROC_fold = np.zeros([NumFold, 4])
    PRECISION_fold = np.zeros([NumFold, 4])
    F05PPR_fold = np.zeros([NumFold, 4])
    
    valid_Precision = np.zeros([NumFold, 4])
    valid_PPR = np.zeros([NumFold, 4])
    Pred_Prob_test = np.zeros([X_data.shape[0], 6])
    CUTOFF = np.zeros([NumFold, 4])

    kf = StratifiedKFold(n_splits=NumFold, shuffle=True, random_state=times * 10 + 1)
    val = 0
    for train_idx, test_idx in kf.split(X_data, Y_data):

        train_X = X_data[train_idx, :]
        train_Y = Y_data[train_idx]
        test_X = X_data[test_idx, :]
        test_Y = Y_data[test_idx]
        Pred_Prob_test[test_idx, 0] = test_Y
        Pred_Prob_test[test_idx, 1] = val + 1

        # Filter out genes using T-test
        DEGs = Filter_Ttest(train_X, train_Y, significant_level = 0.1)
        train_X = train_X[:, DEGs]
        test_X = test_X[:, DEGs]

        if Method == 'RN_SMOTE':
            smote = SMOTE(random_state=2020)
            smox, smoy = smote.fit_sample(train_X, train_Y)
            isSMOTE = 1
        else:
            smox, smoy = train_X, train_Y
            isSMOTE = 0

        Clf_name = ['LASSO', 'RF', 'XGB', 'SVM']
        Classifiers = {'LASSO': LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=2020),
                       'RF': RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=2020),
                       'XGB': XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=5, seed=2020),
                       'SVM': SVC(C=0.01, probability=True, kernel='linear', random_state=2020, max_iter=10000)}
        Parameters = {'LASSO': {'C': [.01, .05, .1, .5, 1.0, 5.0, 10.0], 'fit_intercept': [True, False]},
                      'RF': {'criterion': ['gini', 'entropy'], 'max_depth': sp_randint(1,11)},
                      'XGB': {'learning_rate': [.01, .05, .1, .5], 'max_depth': sp_randint(1,11),
                              'min_child_weight': [1, 2, 3]},
                      'SVM': {'C': [.0001, .001, .005, .01, .05, .1, .5, 1.0, 3.0, 5.0, 10.0]}}
        # -----------------------------------------------------------
        # Feature Selection
        for idx_clf in range(4):
            os.system("echo 'Starting " + Clf_name[idx_clf] + "...'")
            estimator = Classifiers[Clf_name[idx_clf]]
            if NumOfFea == 0:
                selector = RFECV(estimator, step=0.2, cv=3, scoring=myscorer)
            else:
                selector = RFE(estimator, NumOfFea, step=0.2)

            selector = selector.fit(smox, smoy)
            smox_reduced = selector.transform(smox)
            test_X_reduced = selector.transform(test_X)

            cv_parameter = Parameters[Clf_name[idx_clf]]
            searcher = RandomizedSearchCV(estimator, cv_parameter, scoring=myscorer,
                                          random_state=2020).fit(smox_reduced, smoy)
            Pred_test = searcher.predict(test_X_reduced)
            Prob_test = searcher.predict_proba(test_X_reduced)[:, 1]
            F05score_fold[val, idx_clf] = fbeta_score(test_Y, Pred_test, 1.0)
            percentile_CUTOFF, valid_Precision[val, idx_clf], valid_PPR[val, idx_clf] = \
                PPR_percentile_findCutoff(smox_reduced, smoy, searcher.best_estimator_, target_Precision, 10, isSMOTE)
            PPR_fold[val, idx_clf] = recall_score(test_Y, Prob_test >= np.percentile(Prob_test, percentile_CUTOFF))
            PRECISION_fold[val, idx_clf] = \
                precision_score(test_Y, Prob_test >= np.percentile(Prob_test, percentile_CUTOFF))
            F05PPR_fold[val, idx_clf] = \
                fbeta_score(test_Y, Prob_test >= np.percentile(Prob_test, percentile_CUTOFF), beta=1.0)
            AUROC_fold[val, idx_clf] = roc_auc_score(test_Y, Prob_test)
            Pred_Prob_test[test_idx, idx_clf + 2] = Prob_test
            CUTOFF[val, idx_clf] = percentile_CUTOFF
        val += 1
    return F05score_fold, PPR_fold, PRECISION_fold, F05PPR_fold, AUROC_fold, valid_Precision, valid_PPR, \
           Pred_Prob_test, CUTOFF


# -----------------------------
'''
    :param times: Number of repetitions
    :param Clinical: Which variable to predict. Should be index of columns.
    :param NumOfFea: Number of genes to be selected
    :param Criteria: the score used in building model, Possible options: FSCORE, PPR90, PPR95
    :param Method: RN_only, RN_SMOTE, no_RN_no_SMOTE
    :param NumFold: number of folds in k-fold CV
    :return: F05score_fold, PPR_fold, PRECISION_fold, F05PPR_fold, AUROC_fold, valid_Precision
'''
# Clinical, NumOfFea, Criteria, Method, dateTag = sys.argv[1:]

import time
start_time = time.time()

Clinical, NumOfFea, Criteria, Method, dateTag = sys.argv[1:]

F1score_fold, PPR_fold, PRECISION_fold, F1PPR_fold, AUROC_fold, valid_Precision, valid_PPR, Pred_Prob_test, CUTOFF = \
    RepeatCV(1, int(Clinical), int(NumOfFea), Criteria, Method, NumFold = 10)
print(valid_Precision)
dir_name = str(Clinical) + '_' + Method + '_' + dateTag + '_' + Criteria

resultDir = "/gpfs/home/dz16e/Reusability/NewExperiment/Result"
path_name = resultDir + '/' + dir_name
print(dir_name, path_name)

try:
	os.makedirs(path_name)
except:
	print(path_name + "already exist")
	pass

for i in ['F1score_fold', 'PPR_fold', 'PRECISION_fold', 'F1PPR_fold', 'AUROC_fold', 'valid_Precision',
          'valid_PPR', 'Pred_Prob_test', 'CUTOFF']:
    file_name = str(Clinical) + '_' + str(NumOfFea) + '_' + i + '.txt'
    np.savetxt(path_name + '/' + file_name, eval(i), fmt='%8.6f', delimiter=' ')

print("--- %s seconds ---" % (time.time() - start_time))
