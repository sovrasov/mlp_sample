from utils.data import *
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main():
    objects, labels, ids =  load_data('./data/train.csv')
    scaler = StandardScaler()
    scaler.fit(objects)
    print(scaler.mean_, scaler.var_)
    objects = scaler.transform(objects)
    xTrain, yTrain, xTest, yTest = split_dataset(objects, labels, testRatio=.4, seed=1)

    n_epochs = 2
    batch_size = 200
    num_iters = int(float(len(xTrain)) / batch_size * n_epochs)
    print('Max num_iters {}'.format(num_iters))
    clf = MLPClassifier(hidden_layer_sizes=(10), alpha=0.000, activation='relu',\
        solver='sgd', learning_rate='adaptive', learning_rate_init=0.1, verbose=True,\
        momentum=0.9, batch_size=batch_size, max_iter=num_iters, tol=1e-8)

    print('Training...')
    clf.fit(xTrain, yTrain)
    print('Testing...')
    score = clf.score(xTest, yTest)
    print('Test score = ' + str(score))

    score = clf.score(xTrain, yTrain)
    print('Train score = ' + str(score))

if __name__ == '__main__':
    main()
