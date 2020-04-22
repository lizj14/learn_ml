import numpy as np
from utils import *
from model import Model

class LogisticRegression(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weights = None
        self.bias = None
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.l1 = kwargs.get('l1', 0.001)

    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        data = kwargs['data']
        self.weights =  np.random.random((data.shape[1]+1,))
        # self.bias = np.random.random((1))

    def train(self, data, label):
        if not self.init:
            self.init_model(data=data)
        data = np.array(data)
        # data = np.concatenate([data, np.ones((data.shape[0]))], axis=1)
        data = self.add_bias(data)
        self.modify_weight(data, label)
        predict = self.calculate(data)
        # return np.mean((predict==label).astype('float'))
        return clf_precision(predict, label)

    # The data has concatenate with bias '1'
    def calculate(self, data):
        return sigmoid(np.dot(data,self.weights))

    def add_bias(self, data):
        # print('d: {}; one: {}'.format(data.shape, np.ones(data.shape[0]).shape))
        return np.concatenate([data, 
            np.ones((data.shape[0])).reshape(-1,1)], axis=1)

    def modify_weight(self, data, label):
        num_sample = len(label)
        label = np.array(label).reshape(-1,1)
        value = self.calculate(data).reshape(-1,1)
        gradient = np.sum(-label * data + value * data, axis=0)
        gradient *= self.learning_rate / num_sample 
        gradient += self.l1 * self.weights * self.learning_rate
        self.weights -= gradient
        
    def predict(self, data):
        data =self.add_bias(np.array(data))
        return self.calculate(data)

if __name__ == '__main__':
    data = dict_dataset['titanic']
    LogisticRegression(learning_rate=0.1, iter=20).work(data)
