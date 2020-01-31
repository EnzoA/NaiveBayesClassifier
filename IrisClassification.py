import pandas as pd
from NaiveBayesClassifier import DiscreteNaiveBayesClassifier
from NaiveBayesClassifier import GaussianNaiveBayesClassifier

df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df = df.sample(frac=1)

df.loc[df.species == 'setosa', 'species'] = 0
df.loc[df.species == 'versicolor', 'species'] = 1
df.loc[df.species == 'virginica', 'species'] = 2

x = df.loc[:, df.columns != 'species']
y = df.loc[:, df.columns == 'species']

samples_count = df.shape[0]
index = int(0.8 * samples_count)
x_train = x.iloc[:index].values
x_test = x.iloc[index:].values
y_train = y.iloc[:index].values
y_test = y.iloc[index:].values

'''
discrete_clf = DiscreteNaiveBayesClassifier(buckets_number=5)
discrete_clf = discrete_clf.fit(x_train, y_train)
y_pred = discrete_clf.predict(x_test)
accuracy = discrete_clf.accuracy_score(y_test, y_pred)

print('The accuracy is: {}'.format(accuracy))
'''

gaussian_clf = GaussianNaiveBayesClassifier()
gaussian_clf = gaussian_clf.fit(x_train, y_train)
y_pred = gaussian_clf.predict(x_test)
accuracy = gaussian_clf.accuracy_score(y_test, y_pred)

print('The accuracy is: {}'.format(accuracy))

for y_true_i, y_pred_i in zip(y_test, y_pred):
    print('Expected: {0} - Predicted {1}'.format(y_true_i[0], y_pred_i))