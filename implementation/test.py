from utils import *

def test_read_data(name):
    data = dict_dataset[name]
    # print(data.train_data())
    print(data.train_data().shape)
    print(data.test_data().shape)

if __name__ == '__main__':
    test_read_data('titanic')
