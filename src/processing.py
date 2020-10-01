from sklearn.datasets import load_breast_cancer
import math
from sklearn.model_selection import KFold
import numpy as np


def under_sampling(X, y):
    '''
    input: numpy array
    return: list
    '''
    bool_act = y==1
    n_data_sets = math.ceil(len(y[~bool_act])/len(y[bool_act]))
    if n_data_sets==1:
        print('This dataset is balanced data.')
        return X, y
    elif n_data_sets>1:
        print('This dataset is inbalanced data. Run under sampling....')
        print(f'n_datasets = {n_data_sets}')
        x_nega = X[~bool_act]
        y_nega = y[~bool_act]
        x_posi = X[bool_act]
        y_posi = y[bool_act]
        print(f'positive samples = {len(y_posi)}')
        kf = KFold(n_splits=n_data_sets, random_state=1719, shuffle=True)
        idxes = [idx for _, idx in kf.split(x_nega, y_nega)]
        under_sampling_X = [np.concatenate([x_nega[idx], x_posi]) for idx in idxes]
        under_sampling_y = [np.concatenate([y_nega[idx], y_posi]) for idx in idxes]
        print(f'Total {n_data_sets} datasets were generated.')
        return under_sampling_X, under_sampling_y


# def filL_nan(X, y, option):
#     '''
#     Input: array or list of array
#     Return: array or list of array
#     '''
#     flag = type(a).__name__
#     if flag == 'ndarray':
#         if option == 'zero':
#         elif option == 'mean':
#         elif option == 'median':
#     elif flag == 'list':
#         if option == 'zero':
#         elif option == 'mean':
#         elif option == 'median':
            
            
            
            