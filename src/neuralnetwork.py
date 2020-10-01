from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import QuantileTransformer
from keras.utils import np_utils


class NNClassifier:
    '''
    Usage:
    clf = NNClassifier(**params)
    history = clf.fit(
    X_train,
    y_train,
    X_valid,
    y_valid,
    early_stopping_rounds
    )
    '''
    
    def __init__(self, input_shape=1024, input_dropout=0.2, hidden_layers=1, hidden_units=64, hidden_dropout=0.2,
                 batch_norm="none", learning_rate=0.05, batch_size=64, epochs=10000):
        self.input_shape = int(input_shape) # layer param
        self.input_dropout = input_dropout # layer param
        self.hidden_layers = int(hidden_layers) # layer param
        self.hidden_units = int(hidden_units) # layer param
        self.hidden_dropout = hidden_dropout # layer param
        self.batch_norm = batch_norm # layer param
        self.learning_rate = learning_rate # optimizer param
        self.batch_size = int(batch_size) # fit param
        self.epochs = int(epochs) # fit param
        
    def fit(self, X_train, y_train, X_valid, y_valid, early_stopping_rounds):
        # Data standardization
        self.transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        X_train = self.transformer.fit_transform(X_train)
        X_valid = self.transformer.transform(X_valid)
        # layers
        self.model = Sequential()
        self.model.add(Dropout(self.input_dropout, input_shape=(self.input_shape,)))
        for i in range(self.hidden_layers):
            self.model.add(Dense(self.hidden_units))
            if self.batch_norm == 'before_act':
                self. model.add(BatchNormalization())
            self.model.add(ReLU())
            self.model.add(Dropout(self.hidden_dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        # Optimazer
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, decay=0.)
        # Compile
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        # train
        early_stopping = EarlyStopping(patience=early_stopping_rounds, restore_best_weights=True)
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stopping]
        )
        return self.history
    
    def predict(self, x):
        x = self.transformer.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred

    def get_model(self):
        return self.model