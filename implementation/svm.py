import numpy as np
from utils import *
from model import Model
import _svm
from sklearn.utils import check_array, check_random_state


# Attention: only code for understanding the algorithm. Has not debug.
class SupportVectorMachineSoft(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.C = kwargs.get('C', 1)
        self.kernel = dict_kernel[kwargs.get('kernel', 'linear')]
        self.tolerance = kwargs.get('tolerance', 1e-4)
        self.alphatol = kwargs.get('alphatol', 1e-7)
        self.smo_iter = kwargs.get('smo_iter', 1000)
        self.halt_pass = kwargs.get('halt_pass', 10)
        self.random_state = kwargs.get('random', None)
        # support vector
        self.support_vector = None
        

    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        # This is the kernel matrix.
        self.K = self._kernel_matrix(data) 
       
    def kernel_matrix(self, data):
        num_sample, num_feature = data.shape
        K = np.zeros((num_sample, num_feature))
        # non-vector version from https://github.com/ajtulloch/svmpyi
        for i, x_i in enumerate(data):
            for j, x_j in enumerate(data):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def train(self, data, label):
        if not self.init:
            self.init_model(data=data)
        self.support_vector = check_array(data)
        self.label = check_array(label, ensure_2d=False)

        random_state = check_random_state(self.random_state)

        self.K = self.kernel_matrix(data)
        self.dual_coef = np.zeros(data.shape[0])
        self.intercept = _svm.smo(self.K, label, self.dual_coef, self.C,
            random_state, self.tolerance, self.halt_pass, self.smo_iter)

        if_support_vectors = np.nonzero(self.dual_coef)
        self.dual_coef = self.dual_coef[if_support_vectors]
        self.support_vector = data[if_support_vectors]
        self.label = label[if_support_vectors]

        # Here needs to add the predict result


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

class Kernel:
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma):
        return lambda x, y: np.exp(-np.linalg.norm(x-y)**2 / (2*sigma**2) ) 

    @staticmethod
    def poly(degree):
        return lambda x, y: np.inner(x, y) ** degree

    @staticmethod
    def laplace(sigma):
        return lambda x, y: np.exp(-np.linalg.norm(x-y) / sigma)
    
    @staticmethod
    def sigmoid(beta, theta):
        return lambda x, y: np.tanh(beta * np.inner(x, y) + theta)

dict_kernel = {'linear': Kernel.linear(),
        'gaussian': Kernel.gaussian(1.0),
        }

if __name__ == '__main__':
    data = dict_dataset['titanic']
    LogisticRegression(learning_rate=0.1, iter=20).work(data)
