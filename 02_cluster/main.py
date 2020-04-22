import numpy as np
import random as rd
from kmeans import KMeans

def generate_values(dimension, number):
    return np.random.random((number, dimension))

def generate_centered_values(dimension, number, core, 
        num_radius=10, loc_radius=0.1, test=False):
    center = np.random.random((core, dimension))
    matrix_list = []
    for loc in center:
        num_cluster = int(number + (rd.random()-0.5)*2*num_radius)
        dots = np.array([loc] * num_cluster)
        dots += np.random.random(dots.shape) * loc_radius
        matrix_list.append(dots)
    centered_values = np.concatenate(matrix_list)
    if not test:
        np.random.shuffle(centered_values)
    return centered_values, center
        

def test_generate():
    print(generate_values(2, 100))

def test_center():
    print(generate_centered_values(2, 5, 3, num_radius=2, test=True))

def test_kmeans():
    k = KMeans(10, test=True)
    dots, centers = generate_centered_values(2, 100, 10)
    print('real: {}'.format(centers))
    get = k.cluster(dots, 2, 10)
    print('get: {}'.format(get))

if __name__ == '__main__':
    # test_generate()
    # test_center()
    test_kmeans()
