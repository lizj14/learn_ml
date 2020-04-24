import numpy as np
import pandas as pd
from sklearn import preprocessing

class DateSet:
    def __init__(self, name, train_data_path, test_data_path, label, exclude):
        self.name = name
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.label = label
        self.exclude = set(exclude)

    def train_data(self):
        pass

    def test_data(self):
        pass

    def feature_list(self, columns):
        return [f for f in columns if f not in self.exclude]

"""
There is a fault in the process_data. The train_data and test_data are not labelled at the same time. This may led to fault. However, the program is only used for learning, thus the data is not test, but just train and valid. And the valid data is processed together with that of trainning.
The bug will be fixed in the future... or never?
"""
class NormalDataSet(DateSet):
    def __init__(self, name, train_data_path, test_data_path, label, exclude):
        super().__init__(name, train_data_path, test_data_path, label, exclude)

    def train_data(self):
        return self.process_data(self.train_data_path, self.label)

    def test_data(self):
        return self.process_data(self.test_data_path, self.label)

    def process_data(self, data, label):
        pd_data = pd.read_csv(data)
        for feature in pd_data.columns:
            if feature == self.label:
                continue
            if pd_data[feature].dtype == 'object':
                pd_data[feature] = pd_data[feature].fillna('missing')
                le = preprocessing.LabelEncoder()
                le = le.fit(pd_data[feature].values)
                pd_data[feature] = le.transform(pd_data[feature])
            elif pd_data[feature].dtype == 'float64':
                avg = np.mean(pd_data[feature][(1-pd_data[feature].isna()
                    ).astype('bool')])
                pd_data[feature] = pd_data[feature].fillna(avg)
            pd_data[feature] = preprocessing.scale(pd_data[feature])
        return pd_data

class DiscreteDataset(NormalDataSet):
    def __init__(self, name, train_data_path, test_data_path, label,
            discrete, continuous, exclude):
        super().__init__(name, train_data_path, test_data_path, label, exclude)
        self.discrete = discrete
        self.continuous = continuous
        # self.exclude = set(exclude)

    def feature_list(self, columns):
        return self.discrete + self.continuous

    def process_data(self, data, label):
        pd_data = pd.read_csv(data)
        for feature in pd_data.columns:
            if feature == self.label or feature in self.exclude:
                continue
            if pd_data[feature].dtype == 'object':
                pd_data[feature] = pd_data[feature].fillna('missing')
                le = preprocessing.LabelEncoder()
                le = le.fit(pd_data[feature].values)
                pd_data[feature] = le.transform(pd_data[feature])
            # elif pd_data[feature].dtype == 'float64':
            elif feature in self.continuous:
                avg = np.mean(pd_data[feature][(1-pd_data[feature].isna()
                    ).astype('bool')])
                pd_data[feature] = pd_data[feature].fillna(avg)
                pd_data[feature] = preprocessing.scale(pd_data[feature])
        return pd_data

# change (0, 1) -> (-1, +1)
class SVMDataSet(NormalDataSet):
    def __init__(self, name, train_data_path, test_data_path, label):
        super().__init__('svm', train_data_path, test_data_path, label)

    def process_data(self, data, label):
        pd_data = super().process_data(data, label)
        pd_data[label] = (feature_label - 0.5) * 2
        return pd_data

dict_dataset = {
        'titanic': NormalDataSet('titanic', 'titanic/train.csv',
            'titanic/test.csv', 'Survived', ['PassengerId', 'Name']),
        'titanic_dis': DiscreteDataset('titanic', 'titanic/train.csv',
            'titanic/test.csv', 'Survived', ['Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket','Cabin', 'Embarked'], ['Age', 'Fare'], ['PassengerId','Name'] ),
        }

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def precision(a, b):
    # return np.mean((a==b).astype('float'))
    # print('a: {}; b: {}'.format(a[:5], b[:5]))
    # print(a==b)
    # print(np.mean(a==b))
    return np.mean(a==b)

def clf_precision(a, b):
    a = np.array((a > 0.5)).astype('int').astype('float')
    b = np.array(b).astype('float')
    return precision(a, b)

