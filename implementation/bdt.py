import numpy as np
from utils import *
from model import Model
import math

class NaiveBayesClf(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        self.label_possiblity = {}
        self.attribute_possibility = {}
        self.continuous_mean = {}
        self.continuous_std = {}

    def train(self, data, label):
        if not self.init:
            self.init_model(data=data)
        num_data = data.shape[0]
        label_unique = label.unique()
        num_label = len(label_unique)
        dict_label_dis = {}
        for label_now in label_unique:
            frequence = np.sum( label == label_now )
            dict_label_dis[label_now] = data[label_now == label]
            self.label_possiblity[label_now]= (frequence+1)/(num_data+num_label)
            
        label_column = self.data_set.label
        for feature in self.data_set.discrete:
            value_list = data[feature].unique()
            dict_feature = {}
            # distribute = data.groupby(feature).count()[label_column].to_dict()
            # select the data of the same label
            for label_now, sub_data in dict_label_dis.items():
                dict_label_now = {}
                for value in value_list:
                    frequence = np.sum(sub_data[feature] == value)
                    dict_label_now[value] = (frequence+1) / \
                        (sub_data.shape[0]+len(value_list))                    
                dict_feature[label_now] = dict_label_now

            self.attribute_possibility[feature] = dict_feature

        for feature in self.data_set.continuous:
            dict_mean = {}
            dict_std = {}
            for label_now, sub_data in dict_label_dis.items():
                avg_value = np.mean(sub_data[feature])
                std_value = np.std(sub_data[feature])
                dict_mean[label_now] = avg_value
                dict_std[label_now] = std_value
            self.continuous_mean[feature] = dict_mean
            self.continuous_std[feature] = dict_std

        result = self.calculate(data)
        return clf_precision(result, label)

    def calculate(self, data):
        data = data.copy(deep=True)
        num_label = len(self.label_possiblity)
        # num_feature = len(self.data_set.discrete) + len(
        #         self.data_set.continuous) + 1
        probability_matrix = np.zeros((data.shape[0], num_label))
        label_list = list(self.label_possiblity.keys())
        label_no = {}
        for i in range(0, len(label_list)):
            probability_matrix[:, i] += np.log(
                    self.label_possiblity[label_list[i]])
            label_no[label_list[i]] = i

        small_value = 1.0 / data.shape[0]
        for feature in self.data_set.discrete:
            for i in range(0, len(label_list)):
                probability_matrix[:, i] += np.log(data[feature].apply(
                    lambda x: self.attribute_possibility[feature][label_list[i]
                        ].get(x, small_value)))
        for feature in self.data_set.continuous:
            for i in range(0, len(label_list)):
                mean = data[feature].apply( 
                        lambda x: self.continuous_mean[feature][label_list[i]])
                std = data[feature].apply(
                        lambda x: self.continuous_std[feature][label_list[i]])
                f_v = 1.0 / (math.sqrt(math.pi * 2) * std)
                f_v *= np.exp(-(data[feature]-mean)**2 / (2*std**2))
                probability_matrix[:, i] += np.log(f_v)

        select = np.argmax(probability_matrix, axis=1)
        f = np.frompyfunc(lambda x: label_no[x], 1, 1)
        return f(select)
       
    def predict(self, data):
        # data =self.add_bias(np.array(data))
        return self.calculate(data)

if __name__ == '__main__':
    data = dict_dataset['titanic_dis']
    NaiveBayesClf(learning_rate=0.1, iter=1).work(data)
