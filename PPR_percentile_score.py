import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import precision_score, recall_score

def PPR_percentile_score(Y, predicted_proba, Precision, Return_cutoff):
    #predicted_proba = predicted_proba[:, 1]
    #precision_array = np.zeros([len(Y)])
    #recall_array = np.zeros([len(Y)])
    precision_array = np.zeros([100])
    recall_array = np.zeros([100])
    percentile_array = np.zeros([100,])
    percentile_CUTOFF = 100  #initialize the cutoff as 100 percentile
    score = 0.0  #initialize the PPR score as 0
    for pct in range(100):
        percentile_array[pct] = pct
        proba_percentile = np.percentile(predicted_proba, q = pct)
        Pred_Y = 1.0 * (predicted_proba>=proba_percentile)
        precision_array[pct] = precision_score(Y, Pred_Y)
        recall_array[pct] = recall_score(Y, Pred_Y)
        if precision_array[pct] >= Precision and percentile_CUTOFF == 100:
            percentile_CUTOFF = pct
            score = recall_array[pct]

    if Return_cutoff == 1:
        return([score,percentile_CUTOFF])
    else:
        return (score)
