import numpy as np

class KMeans:
    def __init__(self, core_number, test=False):
        self.core_number = core_number
        self.iterate_times = 0
        self.max_it = 10
        self.test = test

    def init_center(self, dimension, core_number):
        self.iterate_times = 0
        return np.random.random((dimension, core_number))

    def cluster(self, dots, dimension, core_number=None):
        if core_number is None:
            core_number = self.core_number
        center = self.init_center(core_number, dimension)
        if self.test:
            print('dots: {}; center: {}'.format(dots.shape, center.shape))
        if self.test:
            print('init center: {}'.format(center))

        while self.judge_end(dots, center):
            diff = np.expand_dims(center, 1).repeat(dots.shape[0], 1) - dots
            diff *= diff
            diff = np.sum(diff, axis=-1)
            nearest_id = np.argmin(diff, axis=0)
            nearest_center = center[nearest_id]

            for no in range(0, center.shape[0]):
                points = dots[nearest_id==no]
                if points.shape[0] != 0:
                    center[no] = np.mean(points, axis=0)

            if self.test:
                print(center)

        return center

    def judge_end(self, dots, center):
        if self.iterate_times > self.max_it:
            return False
        else:
            self.iterate_times += 1
            return True
