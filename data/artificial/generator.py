from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from analysis.utils import plot_image_grid


class ArtificialDatasetGenerator:

    @property
    def num_classes(self):
        raise NotImplementedError()

    def generate_sample(self, cls):
        raise NotImplementedError

    def generate_samples(self, num_samples_per_class):
        X = []
        Y = []
        for cls in tqdm(range(self.num_classes)):
            for _ in tqdm(range(num_samples_per_class)):
                X.append(self.generate_sample(cls))
                Y.append(cls)
        X, Y = shuffle(X, Y)
        return np.array(X), np.array(Y)

    def generate_dataset(self, filename, num_train, num_test):
        xtrain, ytrain = self.generate_samples(num_train // self.num_classes)
        xtest, ytest = self.generate_samples(num_test // self.num_classes)
        np.savez_compressed(filename, x_train=xtrain, x_test=xtest, y_train=ytrain, y_test=ytest)

    def show_samples(self, cls, count=4):
        xs = []
        for i in range(count):
            xs.append(self.generate_sample(cls))
        plot_image_grid(xs)
