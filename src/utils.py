import pickle


def load(path):
    with open('{}'.format(path), 'rb') as f:
        item = pickle.load(f)
    return item


def save(path, item):
    with open('{}'.format(path), 'wb') as f:
        pickle.dump(item, f)


class Paramset:
    '''
    # Usage
    paramset = Paramset(XGBRegressor())
    paramset.swiching_lr('params_search')
    XGB_PARAMS = paramset.generate_params()
    XGB_PARAMS
    >> {'learing_rate': 0.05,
        'n_estimators': 100000,
        'max_depth': 9,
        'subsample': 0.65,
        'colsample_bytree': 0.65,
        'gamma': 1,
        'min_child_weight': 10,
        'random_state': 1112,
        'n_jobs': -1}
    '''

    def __init__(self, model):
        self.PARAMS = {}
        self.model = model
        self.model_type = type(self.model).__name__

    def generate_params(self):
        if self.model_type == "LGBMClassifier":
            self.PARAMS.update(
                {
                    'n_estimators': 100000,
                    'max_depth': -1,
                    'num_leaves': 31,
                    'subsample': 0.65,
                    'colsample_bytree': 0.65,
                    'bagging_freq': 10,
                    'min_child_weight': 10,
                    'min_child_samples': 10,
                    'min_split_gain': 0.01,
                    'random_state': 1112,
                    'n_jobs': -1
                }
            )
            return self.PARAMS
        elif self.model_type == "XGBClassifier":
            self.PARAMS.update(
                {
                    'n_estimators': 100000,
                    'max_depth': 9,
                    'subsample': 0.65,
                    'colsample_bytree': 0.65,
                    'gamma': 1,
                    'min_child_weight': 10,
                    'random_state': 1112,
                    'n_jobs': -1
                }
            )
            return self.PARAMS
        elif self.model_type == "NNClassifier":
            self.PARAMS.update(
                {
                    'input_shape':1024,
                    "input_dropout":0.2,
                    "hidden_layers":1,
                    'hidden_units':64,
                    'hidden_dropout':0.2,
                    'batch_norm':'before_act',
                    'batch_size':64,
                    'epochs':10000
                }
            )
            return self.PARAMS
            
    def swiching_lr(self, swich):
        if swich == 'train':
            self.PARAMS.update({'learning_rate': 1e-3})
            return self.PARAMS
        elif swich == 'params_search':
            self.PARAMS.update({'learning_rate': 0.05})
            return self.PARAMS