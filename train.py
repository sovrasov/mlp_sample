from utils.data import *
from mlp.mlp import MLP

import numpy as np
from sklearn.neural_network import MLPClassifier

def main():
    objects, labels, ids = load_data('./data/train.csv')
    normalizer = DataNormalizer()
    normalizer.fit(objects)
    objects = normalizer.transform(objects)

    xTrain, yTrain, xTest, yTest = split_dataset(objects, labels, testRatio=.2, seed=1)

    n_epochs = 2
    batch_size = 200
    num_iters = int(float(len(xTrain)) / batch_size * n_epochs)
    print('SGD max iters: {}'.format(num_iters))

    clf = MLP(lr=0.1, bs=batch_size, momentum=0.9, verbose=True, max_iters=num_iters, eps=1e-8)

    #clf = MLPClassifier(hidden_layer_sizes=(10), alpha=0.000, activation='relu',\
    #    solver='sgd', learning_rate='adaptive', learning_rate_init=0.1, verbose=True,\
    #    momentum=0.9, batch_size=batch_size, max_iter=num_iters, tol=1e-8)

    print('Training...')
    clf.fit(xTrain, yTrain)
    print('Testing...')
    score = clf.score(xTest, yTest)
    print('Test score = ' + str(score))

    score = clf.score(xTrain, yTrain)
    print('Train score = ' + str(score))

    print('Writing a submission file...')
    test_objects, test_labels, test_ids = load_data('./data/test.csv')
    test_objects = normalizer.transform(test_objects)
    predictions = clf.predict(test_objects)
    save_predictions('submission.csv', test_ids, predictions)

if __name__ == '__main__':
    main()
