import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
            self.train_rmse = np.array(
                self.model.evals_result_['training']['binary_logloss'])
            self.valid_rmse = np.array(
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
            self.train_rmse = np.array(
                self.model.evals_result_['validation_0']['logloss'])
            self.valid_rmse = np.array(
                self.model.evals_result_['validation_1']['logloss'])

    def predict_proba(self, X):
        if self.model_type == "LGBMClassifier" or "XGBClassifier":
            return self.model.predict_proba(X, ntree_limit=self.best_iteration)

    def get_model(self):
        return self.model

    def get_best_iteration(self):
        return self.best_iteration

    def get_importance(self):
        return self.importance

    def get_learning_curve(self):
        palette = sns.diverging_palette(220, 20, n=2)
        width = np.arange(self.train_rmse.shape[0])
        plt.figure(figsize=(10, 7.32))
        plt.title(
            'Learning_Curve ({})'.format(self.model_type), fontsize=15)
        plt.xlabel('Iterations', fontsize=15)
        plt.ylabel('LogLoss', fontsize=15)
        plt.plot(width, self.train_rmse, label='train_logloss', color=palette[0])
        plt.plot(width, self.valid_rmse, label='valid_logloss', color=palette[1])
        plt.legend(loc='upper right', fontsize=13)
        plt.show()