import numpy as np
from scipy.stats import ttest_ind, levene

def Filter_Ttest(Predictors, labels, significant_level = 0.05):
    '''
    Perform Welch two-sample t-test to find differentially expressed genes.
    :param Predictors: N patients by P genes matrix
    :param labels: array contains two classes
    :param significant_level: the significant level
    :return: index for significant genes
    '''

    Classes = np.unique(labels)
    if len(Classes) > 2:
        print('Number of classes is greater than 2!')

    Sample1 = Predictors[labels == Classes[0], :]
    Sample2 = Predictors[labels == Classes[1], :]
    DEGs = []
    for i in range(Sample1.shape[1]):
        Var_test = levene(Sample1[:, i], Sample2[:, i])
        Var_test_pvalue = Var_test[1]
        if Var_test_pvalue < 0.05:
            T_Test = ttest_ind(Sample1[:, i], Sample2[:, i], equal_var=False)
        else:
            T_Test = ttest_ind(Sample1[:, i], Sample2[:, i], equal_var=True)
        T_Test_pvalue = T_Test[1]
        if T_Test_pvalue < significant_level:
            DEGs.append(i)
    return DEGs






