from utils import *
# The main method.
from lr import LogisticRegression

def run_model(model, data_set):
    model.work(data_set)

def run_lr():
    run_model(LogisticRegression(), dict_dataset['titanic'])

if __name__ == '__main__':
    pass
