from processing import under_sampling
from trainer import Trainer, cv_and_emsemble_predict
from optimizer import Objective, optuna_search
from utils import Paramset, load, save
from neuralnetwork import NNClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def Optimizer(n_trials, n_jobs, random_state):
    LGB_PARAMS = []
    XGB_PARAMS = []
    NN_PARAMS = []
    if PIPELINE_PARAMS['optimize']==True:
        print('Run Optimizer.')
        flag_inputtype = type(X).__name__
        if flag_inputtype == 'list':
            print(f' Data type of X is list type.')
            print(f' Running optimization for all data sets. Please waite....')
            for x, y in tqdm(zip(X, Y)):
                obj_lgb = Objective(LGBMClassifier(), x, y)
                obj_xgb = Objective(XGBClassifier(), x, y)
                obj_nn = Objective(NNClassifier(), x, y)
                if PIPELINE_PARAMS['LGB']==True:
                    lgb_params = optuna_search(obj_lgb, n_trials, n_jobs, random_state)
                    LGB_PARAMS.append(lgb_params)
                if PIPELINE_PARAMS['XGB']==True:
                    xgb_params = optuna_search(obj_xgb, n_trials, n_jobs, random_state)
                    XGB_PARAMS.append(xgb_params)
                if PIPELINE_PARAMS['NN']==True:
                    nn_params = optuna_search(obj_nn, n_trials, n_jobs, random_state)
                    NN_PARAMS.append(nn_params)  
        elif flag_inputtype == 'ndarray':
            print(f' Data type of X is ndarray.')
            print(f' Running optimization for a data sets.')
            obj_lgb = Objective(LGBMClassifier(), X, Y)
            obj_xgb = Objective(XGBClassifier(), X, Y)
            obj_nn = Objective(NNClassifier(), X, Y)
            if PIPELINE_PARAMS['LGB']==True:
                lgb_params = optuna_search(obj_lgb, n_trials, n_jobs, random_state)
                LGB_PARAMS.append(lgb_params)
            if PIPELINE_PARAMS['XGB']==True:
                xgb_params = optuna_search(obj_xgb, n_trials, n_jobs, random_state)
                XGB_PARAMS.append(xgb_params)
            if PIPELINE_PARAMS['NN']==True:
                nn_params = optuna_search(obj_nn, n_trials, n_jobs, random_state)
                NN_PARAMS.append(nn_params)
    return LGB_PARAMS, XGB_PARAMS, NN_PARAMS


def Ensembler(n_splits, early_stopping_rounds, random_state):
    LGB_PREFDS = []
    XGB_PREFDS = []
    NN_PREFDS = []
    flag_inputtype = type(X).__name__
    if flag_inputtype == 'list':
        for i, x, y in zip(np.arange(len(X)), X, Y):
            print('-----------------------------------------------------------------------------------------------------')
            if len(LGB_PARAMS) != 0:
                print('LGB params were optimized')
                lgb_params.update(LGB_PARAMS[i])
                lgb_params['min_child_samples'] = int(LGB_PARAMS[i]['min_child_samples'])
            else:
                print('LGB params were not optimized')
            if len(XGB_PARAMS) != 0:
                print('XGB params were optimized')
                xgb_params.update(XGB_PARAMS[i])
            else:
                print('XGB params were not optimized')
            if len(NN_PARAMS) != 0:
                nn_params.update(NN_PARAMS[i])
                nn_params['hidden_units'] = int(NN_PARAMS[i]['hidden_units'])
                nn_params['batch_size'] = int(NN_PARAMS[i]['batch_size'])
                print('NN params were optimized')
            else:
                print('NN params were not optimized')         
            if PIPELINE_PARAMS['LGB']==True:
                print('-----------------------------------------------------------------------------------------------------')
                print(f'LGB: {lgb_params}')
                print('-----------------------------------------------------------------------------------------------------')
                va_pred_lgb, te_preds_lgb = cv_and_emsemble_predict(
                    LGBMClassifier(**lgb_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                LGB_PREFDS.append([va_pred_lgb, te_preds_lgb])
            if PIPELINE_PARAMS['XGB']==True:
                print('-----------------------------------------------------------------------------------------------------')
                print(f'XGB: {xgb_params}')
                print('-----------------------------------------------------------------------------------------------------')
                va_pred_xgb, te_preds_xgb = cv_and_emsemble_predict(
                    XGBClassifier(**xgb_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                XGB_PREFDS.append([va_pred_xgb, te_preds_xgb])
            if PIPELINE_PARAMS['NN']==True:
                nn_params['input_shape'] = x.shape[1]
                print('-----------------------------------------------------------------------------------------------------')
                print(f'NN: {nn_params}')
                print('-----------------------------------------------------------------------------------------------------')
                va_pred_nn, te_preds_nn = cv_and_emsemble_predict(
                    NNClassifier(**nn_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                NN_PREFDS.append([va_pred_nn, te_preds_nn])
    elif flag_inputtype == 'ndarray':
        # parameters 
        if len(LGB_PARAMS) != 0:
            print('LGB params were optimized')
            lgb_params.update(LGB_PARAMS[0])
            lgb_params['min_child_samples'] = int(LGB_PARAMS[0]['min_child_samples'])
        else:
            print('LGB params were not optimized')
        if len(XGB_PARAMS) != 0:
            print('XGB params were optimized')
            xgb_params.update(XGB_PARAMS[0])
        else:
            print('XGB params were not optimized')
        if len(NN_PARAMS) != 0:
            nn_params.update(NN_PARAMS[0])
            nn_params['hidden_units'] = int(NN_PARAMS[0]['hidden_units'])
            nn_params['batch_size'] = int(NN_PARAMS[0]['batch_size'])
            print('NN params were optimized')
        else:
            print('NN params were not optimized')
        # training
        if PIPELINE_PARAMS['LGB']==True:
            print('-----------------------------------------------------------------------------------------------------')
            print(f'LGB: {lgb_params}')
            print('-----------------------------------------------------------------------------------------------------')
            va_pred_lgb, te_preds_lgb = cv_and_emsemble_predict(
                LGBMClassifier(**lgb_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            LGB_PREFDS.append([va_pred_lgb, te_preds_lgb])
        if PIPELINE_PARAMS['XGB']==True:
            print('-----------------------------------------------------------------------------------------------------')
            print(f'XGB: {xgb_params}')
            print('-----------------------------------------------------------------------------------------------------')
            va_pred_xgb, te_preds_xgb = cv_and_emsemble_predict(
                XGBClassifier(**xgb_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            XGB_PREFDS.append([va_pred_xgb, te_preds_xgb])
        if PIPELINE_PARAMS['NN']==True:
            nn_params['input_shape'] = X.shape[1]
            print('-----------------------------------------------------------------------------------------------------')
            print(f'NN: {nn_params}')
            print('-----------------------------------------------------------------------------------------------------')
            va_pred_nn, te_preds_nn = cv_and_emsemble_predict(
                NNClassifier(**nn_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            NN_PREFDS.append([va_pred_nn, te_preds_nn])
    return LGB_PREFDS, XGB_PREFDS, NN_PREFDS


if __name__ == '__main__':
    PIPELINE_PARAMS = {
        'X_path': '../data/X.csv',
        'y_path': '../data/y.csv',
        'X_test_path': '../data/X_test.csv',
        'under_sampling': True,
        'optimize': True,
        'LGB': True,
        'XGB': False,
        'NN': False,
    }
    ### Optimizier params
    n_trials = 50
    n_jobs = -1
    random_state_opt = 0
    ### Ensembler params
    n_splits = 5
    early_stopping_rounds = 1000
    random_state_ens = 1522
    # data load
    X = np.array(pd.read_csv(PIPELINE_PARAMS['X_path']).iloc[:,1:])
    Y = np.array(pd.read_csv(PIPELINE_PARAMS['y_path']).iloc[:,1:]).flatten()
    X_test = np.array(pd.read_csv(PIPELINE_PARAMS['X_test_path']).iloc[:,1:])
    print(X.shape)
    Y[:200] = 0
    if PIPELINE_PARAMS['under_sampling']==True:
        X, Y = under_sampling(X, Y)
    #  pipeline
    LGB_PARAMS, XGB_PARAMS, NN_PARAMS = Optimizer(n_trials, n_jobs, random_state_opt)
    LGB_PREFDS, XGB_PREFDS, NN_PREFDS = Ensembler(n_splits, early_stopping_rounds, random_state_esn)