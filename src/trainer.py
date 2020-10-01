import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


class Trainer:

    '''
    # Usage
    n_splits = 3
    random_state = 0
    early_stopping_rounds=10
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for tr_idx, va_idx in kf.split(X, y):
        model = Trainer(XGBRegressor(**XGB_PARAMS))
        model.fit(
            X[tr_idx],
            y[tr_idx],
            X[va_idx],
            y[va_idx],
            early_stopping_rounds
        )
        model.get_learning_curve()
    '''

    def __init__(self, model):
        self.model = model
        self.model_type = type(model).__name__
        self.best_iteration = 100
        self.train_rmse = []
        self.valid_rmse = []
        self.importance = []

    
    def fit(self,
            X_train, y_train, X_valid, y_valid,
            early_stopping_rounds):

        eval_set = [(X_train, y_train), (X_valid, y_valid)]

        if self.model_type == "LGBMClassifier":
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric='logloss',
                verbose=False
            )
            self.best_iteration = self.model.best_iteration_
            self.importance = self.model.booster_.feature_importance(
                importance_type='gain')
            self.train_logloss = np.array(
                self.model.evals_result_['training']['binary_logloss'])
            self.valid_logloss = np.array(
                self.model.evals_result_['valid_1']['binary_logloss'])
            self.importance = self.model.feature_importances_

        elif self.model_type == 'XGBClassifier':
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                eval_metric='logloss',
                verbose=0
            )
            self.best_iteration = self.model.best_iteration
            self.importance = self.model.feature_importances_
            self.train_logloss = np.array(
                self.model.evals_result_['validation_0']['logloss'])
            self.valid_logloss = np.array(
                self.model.evals_result_['validation_1']['logloss'])
            
        elif self.model_type == 'NNClassifier':
            history = self.model.fit(
                X_train,
                y_train,
                X_valid,
                y_valid,
                early_stopping_rounds
            )
            self.train_logloss = np.array(history.history['loss'])
            self.valid_logloss = np.array(history.history['val_loss'])
            

    def predict_proba(self, X):
        if self.model_type == "LGBMClassifier":
            return self.model.predict_proba(X, num_iterations=self.best_iteration)[:,1]
        elif self.model_type == "XGBClassifier":
            return self.model.predict_proba(X, ntree_limit=self.best_iteration)[:,1]
        elif self.model_type == 'NNClassifier':
            return self.model.predict(X)
        
        
    def get_model(self):
        if self.model_type == "LGBMClassifier":
            return self.model
        elif self.model_type == "XGBClassifier":
            return self.model
        elif self.model_type == 'NNClassifier':
            return self.model.get_model()

    
    def get_best_iteration(self):
        print(print(f"model type is {self.model_type}"))
        return self.best_iteration

    
    def get_importance(self):
        if self.model_type == "LGBMClassifier":
            return self.importance
        elif self.model_type == "XGBClassifier":
            return self.importance
        elif self.model_type == 'NNClassifier':
            return 'For NNClassifier, feature importance is not callable.'

        
    def get_learning_curve(self):
        palette = sns.diverging_palette(220, 20, n=2)
        width = np.arange(self.train_logloss.shape[0])
        plt.figure(figsize=(10, 7.32))
        plt.title(
            'Learning_Curve ({})'.format(self.model_type), fontsize=15)
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('LogLoss', fontsize=15)
        plt.plot(width, self.train_logloss, label='train_logloss', color=palette[0])
        plt.plot(width, self.valid_logloss, label='valid_logloss', color=palette[1])
        plt.legend(loc='upper right', fontsize=13)
        plt.show()
        
        
def cv_and_emsemble_predict(model, X_train, y_train, X_test, n_splits, early_stopping_rounds, random_state):
    '''
    Return:
    predicted value of validation set,
    and list of prediction values for the test set predicted by each K-kinds of model in the CV 
    Notaion:
    predicted value of the validation set can be use for building stacking model.
    '''
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    VA_IDXES = []
    VA_PREDS = []
    TE_PREDS = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        clf = Trainer(model)
        clf.fit(
            X_train[tr_idx],
            y_train[tr_idx],
            X_train[va_idx],
            y_train[va_idx],
            early_stopping_rounds
        )
        va_pred = clf.predict_proba(X_train[va_idx])
        te_pred = clf.predict_proba(X_test)
        VA_IDXES.append(va_idx)
        VA_PREDS.append(va_pred)
        TE_PREDS.append(te_pred)
    va_idxes = np.concatenate(VA_IDXES)
    order = np.argsort(va_idxes)
    va_preds = np.concatenate(VA_PREDS)
    va_preds = va_preds[order]
    return va_preds, TE_PREDS