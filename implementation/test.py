from utils import *
# import svm

def test_read_data(name):
    data = dict_dataset[name]
    # print(data.train_data())
    print(data.train_data().shape)
    print(data.test_data().shape)
    return data.train_data()

# def test_kernel():
#     s = svm.SupportVectorMachineSoft()
#     # a = [1,2]
#     # b = [2,3,4]
#     a = np.array([[1,2,4], [0,1,1], [2,2,2], [3,1,0]]) 
#     print(s.kernel_matrix(a))

if __name__ == '__main__':
    test_read_data('titanic')
