import numpy as np
from scipy.stats import norm

class GaussianNaiveBayesClassifier:
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

    def accuracy_score(self, y_true, y_pred):
        zipped = zip(y_true, y_pred)
        matches = np.array([(x == y)[0] for x, y in zipped])
        return matches[matches == True].shape[0] / y_true.shape[0]