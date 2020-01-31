import numpy as np
from scipy.stats import norm
from abc import ABC
from abc import abstractmethod

class NaiveBayesClassifier(ABC):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def accuracy_score(self, y_true, y_pred):
        zipped = zip(y_true, y_pred)
        matches = np.array([(x == y)[0] for x, y in zipped])
        return matches[matches == True].shape[0] / y_true.shape[0]

class DiscreteNaiveBayesClassifier(NaiveBayesClassifier):
    def __init__(self, buckets_number=4):
        self.buckets_number = buckets_number

    def fit(self, x, y):
        x = self.discretize(x, self.buckets_number)
        samples_length = x.shape[0]
        
        self.y_values = np.unique(y)
        self.y_probs = dict([(k, y[y == k].shape[0] / samples_length) for k in self.y_values])
        
        buckets_values = np.unique(x)
        x_y = np.concatenate((x, y), axis=1)
        x_y_cols = np.arange(x_y.shape[1])
        self.likelihood = {}
        features_length = x.shape[1]
        for f_i in np.arange(features_length):
            for y_i in self.y_values:
                indexes = (x_y_cols == f_i) | (x_y_cols == x_y.shape[1] - 1)
                this_x_y = x_y[:,  indexes]
                this_x_y = this_x_y[this_x_y[:, 1] == y_i]
                if f_i not in self.likelihood:
                    self.likelihood[f_i] = {}
                # Using Laplace smoothing with alpha == 1
                self.likelihood[f_i][y_i] = dict([(b, (this_x_y[[this_x_y[:, 0] == b]].shape[0] + 1) / (samples_length + features_length))
                                                for b in buckets_values])
        return self

    def predict(self, x):
        x = self.discretize(x, self.buckets_number)
        y_pred = np.ones((x.shape[0], self.y_values.shape[0])) * -1
        features_arr = np.arange(x.shape[1])
        i = 0
        for x_i in x:
            j = 0
            for y_i in self.y_values:
                for f_i in features_arr:
                    likelihoods = self.likelihood[f_i][y_i]
                    if y_pred[i, j] == -1:
                        y_pred[i, j] = self.y_probs[y_i] * likelihoods[x_i[f_i]]
                    else:
                        y_pred[i, j] *= likelihoods[x_i[f_i]]
                j += 1
            i += 1
        return [np.argmax(y_pred[i]) for i in np.arange(x.shape[0])]

    def discretize(self, x, buckets):
        features_length = x.shape[1]
        limits = np.empty((features_length, buckets))
        for i in np.arange(features_length):
            col = x[:, i]
            min_i = col.min()
            max_i = col.max()
            max_min = max_i - min_i
            limits[i] = np.array([min_i + b / (buckets - 1) * max_min for b in np.arange(buckets)])
            x[:, i] = list(map(lambda c: (np.abs(limits[i] - c)).argmin(), col))
        return x

class GaussianNaiveBayesClassifier(NaiveBayesClassifier):
    def fit(self, x, y):
        self.y_values = np.unique(y)
        self.y_probs = dict([(k, y[y == k].shape[0] / x.shape[0]) for k in self.y_values])

        x_y = np.concatenate((x, y), axis=1)
        x_y_cols = np.arange(x_y.shape[1])
        self.likelihood = {}
        for f_i in np.arange(x.shape[1]):
            for y_i in self.y_values:
                if f_i not in self.likelihood:
                    self.likelihood[f_i] = {}
                indexes = (x_y_cols == f_i) | (x_y_cols == x_y.shape[1] - 1)
                this_x_y = x_y[:,  indexes]
                this_x = this_x_y[this_x_y[:, 1] == y_i, 0]
                self.likelihood[f_i][y_i] = (np.mean(this_x), np.std(this_x))

        return self

    def predict(self, x):
        y_pred = np.ones((x.shape[0], self.y_values.shape[0])) * -1
        features_arr = np.arange(x.shape[1])
        i = 0
        for x_i in x:
            j = 0
            for y_i in self.y_values:
                for f_i in features_arr:
                    likelihoods = self.likelihood[f_i][y_i]
                    if y_pred[i, j] == -1:
                        y_pred[i, j] = self.y_probs[y_i] * norm(likelihoods[0], likelihoods[1]).pdf(x_i[f_i])
                    else:
                        y_pred[i, j] *= norm(likelihoods[0], likelihoods[1]).pdf(x_i[f_i])
                j += 1
            i += 1
        return [np.argmax(y_pred[i]) for i in np.arange(x.shape[0])]