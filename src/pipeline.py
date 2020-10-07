from processing import under_sampling, nan_processing
from trainer import Trainer, cv_and_emsemble_predict
from optimizer import Objective, optuna_search
from utils import Paramset, load, save
from neuralnetwork import NNClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def Optimizer(n_trials, n_jobs, random_state):
    LGB_PARAMS, XGB_PARAMS, NN_PARAMS = [], [], []
    flag_inputtype = type(X).__name__
    if flag_inputtype == 'list':
        print(f' Data type of X is list type.')
        print(f' Running optimization for all data sets. Just a moment....')
        for x, y in zip(X, Y):
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
    LGB_PREDS, XGB_PREDS, NN_PREDS = [], [], []
    flag_inputtype = type(X).__name__
    if flag_inputtype == 'list':
        for i, x, y in zip(np.arange(len(X)), X, Y):
            if len(LGB_PARAMS) != 0:
                print('LGB has optimized parameters')
                lgb_params.update(LGB_PARAMS[i])
                lgb_params['min_child_samples'] = int(LGB_PARAMS[i]['min_child_samples'])
            if len(XGB_PARAMS) != 0:
                print('XGB has optimized parameters')
                xgb_params.update(XGB_PARAMS[i])
            if len(NN_PARAMS) != 0:
                nn_params.update(NN_PARAMS[i])
                nn_params['hidden_units'] = int(NN_PARAMS[i]['hidden_units'])
                nn_params['batch_size'] = int(NN_PARAMS[i]['batch_size'])
                print('NN has optimized parameters')        
            if PIPELINE_PARAMS['LGB']==True:
                print(f'LGB: {lgb_params}')
                print('------------------------------------------------------------------------------------------------')
                va_pred_lgb, te_preds_lgb = cv_and_emsemble_predict(
                    LGBMClassifier(**lgb_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                LGB_PREDS.append([va_pred_lgb, te_preds_lgb])
            if PIPELINE_PARAMS['XGB']==True:
                print(f'XGB: {xgb_params}')
                print('------------------------------------------------------------------------------------------------')
                va_pred_xgb, te_preds_xgb = cv_and_emsemble_predict(
                    XGBClassifier(**xgb_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                XGB_PREDS.append([va_pred_xgb, te_preds_xgb])
            if PIPELINE_PARAMS['NN']==True:
                nn_params['input_shape'] = x.shape[1]
                print(f'NN: {nn_params}')
                print('------------------------------------------------------------------------------------------------')
                va_pred_nn, te_preds_nn = cv_and_emsemble_predict(
                    NNClassifier(**nn_params),
                    x,
                    y,
                    X_test,
                    n_splits,
                    early_stopping_rounds,
                    random_state
                )
                NN_PREDS.append([va_pred_nn, te_preds_nn])
    elif flag_inputtype == 'ndarray':
        if len(LGB_PARAMS) != 0:
            print('LGB has optimized parameters')
            lgb_params.update(LGB_PARAMS[0])
            lgb_params['min_child_samples'] = int(LGB_PARAMS[0]['min_child_samples'])
        if len(XGB_PARAMS) != 0:
            print('XGB has optimized parameters')
            xgb_params.update(XGB_PARAMS[0])
        if len(NN_PARAMS) != 0:
            nn_params.update(NN_PARAMS[0])
            nn_params['hidden_units'] = int(NN_PARAMS[0]['hidden_units'])
            nn_params['batch_size'] = int(NN_PARAMS[0]['batch_size'])
            print('NN params were optimized')
        if PIPELINE_PARAMS['LGB']==True:
            print(f'LGB: {lgb_params}')
            print('------------------------------------------------------------------------------------------------')
            va_pred_lgb, te_preds_lgb = cv_and_emsemble_predict(
                LGBMClassifier(**lgb_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            LGB_PREDS.append([va_pred_lgb, te_preds_lgb])
        if PIPELINE_PARAMS['XGB']==True:
            print(f'XGB: {xgb_params}')
            print('------------------------------------------------------------------------------------------------')
            va_pred_xgb, te_preds_xgb = cv_and_emsemble_predict(
                XGBClassifier(**xgb_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            XGB_PREDS.append([va_pred_xgb, te_preds_xgb])
        if PIPELINE_PARAMS['NN']==True:
            nn_params['input_shape'] = X.shape[1]
            print(f'NN: {nn_params}')
            print('------------------------------------------------------------------------------------------------')
            va_pred_nn, te_preds_nn = cv_and_emsemble_predict(
                NNClassifier(**nn_params),
                X,
                Y,
                X_test,
                n_splits,
                early_stopping_rounds,
                random_state
            )
            NN_PREDS.append([va_pred_nn, te_preds_nn])
    return LGB_PREDS, XGB_PREDS, NN_PREDS


if __name__ == '__main__':
    PIPELINE_PARAMS = {
        'X_path': '../data/X.csv',
        'y_path': '../data/y.csv',
        'X_test_path': '../data/X_test.csv',
        'nan_processing':
        False,
#         'drop',
#         'fillzero',
#         'fillmean',
#         'fillmedian',
        'under_sampling': False,
        'optimize': True,
        'use_stored_params': False,
        'LGB': False,
        'XGB': False,
        'NN': True,
    }
    ### Optimizier params
    n_trials = 3
    n_jobs = -1
    random_state_opt = 0
    ### Ensembler params
    n_splits = 5
    early_stopping_rounds = 10
    random_state_ens = 1522
    
    # load datasets
    X = np.array(pd.read_csv(PIPELINE_PARAMS['X_path']).iloc[:,1:])
    Y = np.array(pd.read_csv(PIPELINE_PARAMS['y_path']).iloc[:,1:]).flatten()
    X_test = np.array(pd.read_csv(PIPELINE_PARAMS['X_test_path']).iloc[:,1:])
    if PIPELINE_PARAMS['nan_processing'] != False:
        X_ = np.concatenate([X, X_test], axis=0)
        X = nan_processing(X_, PIPELINE_PARAMS['nan_processing'])[:len(X),:]
        X_test = nan_processing(X_, PIPELINE_PARAMS['nan_processing'])[len(X_test):,:]
        
    # undersampling
    if PIPELINE_PARAMS['under_sampling']==True:
        X, Y = under_sampling(X, Y)

    # optimization
    if PIPELINE_PARAMS['optimize']==True:
        print('Run Optimizer. Just a moment.......')
        LGB_PARAMS, XGB_PARAMS, NN_PARAMS = Optimizer(n_trials, n_jobs, random_state_opt)
        ITEMS = [LGB_PARAMS, XGB_PARAMS, NN_PARAMS]
        NAMES = ['LGB_PARAMS', 'XGB_PARAMS', 'NN_PARAMS']
        for item, name in zip(ITEMS, NAMES):
            save(f'../result/parameters/{name}.binaryfile', item)
    elif PIPELINE_PARAMS['optimize']==False:
        if PIPELINE_PARAMS['use_stored_params']==True:
            print('Training parameters are referred to from /results/parameters/...')
            if PIPELINE_PARAMS['LGB']==True:
                LGB_PARAMS = load('../result/parameters/LGB_PARAMS.binaryfile')
            if PIPELINE_PARAMS['XGB']==True:
                XGB_PARAMS = load('../result/parameters/XGB_PARAMS.binaryfile')
            if PIPELINE_PARAMS['NN']==True:
                NN_PARAMS = load('../result/parameters/NN_PARAMS.binaryfile')
        elif PIPELINE_PARAMS['use_stored_params']==False:
            print('Training parameters are referred to from utils.py')
            LGB_PARAMS, XGB_PARAMS, NN_PARAMS = {}, {}, {}
    
    #  train
    paramset_lgb = Paramset(LGBMClassifier())
    paramset_lgb.swiching_lr('train')
    lgb_params = paramset_lgb.generate_params()
    paramset_xgb = Paramset(XGBClassifier())
    paramset_xgb.swiching_lr('train')
    xgb_params = paramset_xgb.generate_params()
    paramset_nn = Paramset(NNClassifier())
    paramset_nn.swiching_lr('train')
    nn_params = paramset_nn.generate_params()
    LGB_PREDS, XGB_PREDS, NN_PREDS = Ensembler(n_splits, early_stopping_rounds, random_state_ens)
    ITEMS = [LGB_PREDS, XGB_PREDS, NN_PREDS]
    NAMES = ['LGB_PREDS', 'XGB_PREDS', 'NN_PREDS']
    for item, name in zip(ITEMS, NAMES):
        save(f'../result/predictions/{name}.binaryfile', item)
        
    