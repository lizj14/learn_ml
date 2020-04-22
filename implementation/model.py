import numpy as np
from utils import *

class Model:
    def __init__(self, **kwargs):
        self.valid_proportion = 0.2 if not 'valid' in kwargs.keys() else float(
                kwargs['valid'])
        self.iterations = int(kwargs['iter']) if 'iter' in kwargs.keys() else 5
        self.init = False

    def init_model(self, **kwargs):
        self.init = True

    def train(self, data, label):
        pass

    def predict(self, data):
        pass

    def valid(self, data, label):
        result = self.predict(data)
        # return np.mean(result == label)
        return clf_precision(result, label)

    def work(self, data_set):
        data = data_set.train_data()
        # np.shuffle(data)
        test_data = data_set.test_data()
        data['random'] = np.random.random((data.shape[0]))
        train_data = data[data['random'] >= self.valid_proportion]
        valid_data = data[data['random'] < self.valid_proportion]

        # train_data.drop(['random'], axis=1)
        # valid_data.drop(['random'], axis=1)
        set_exclude = set(['random', data_set.label])
        features = [ f for f in train_data.columns if not f in set_exclude ] 
        for i in range(0, self.iterations):
            # np.shuffle(train_data)
            train_ratio = self.train(
                    train_data[features], train_data[data_set.label])
            print('train ratio: %.3f' % train_ratio)
            valid_ratio = self.valid(
                    valid_data[features], valid_data[data_set.label])
            print('valid ratio: %.3f' % valid_ratio)
        predict = self.predict(test_data[features])
        return predict
