import random
import numpy as np
import csv

def load_data(data_path):
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        objects = []
        labels = []
        ids = []
        for row in reader:
            x = [float(row[i]) for i in range(1, len(row) - 1)]
            y = int(row[-1])
            id = int(float(row[0]))
            objects.append(x)
            labels.append(y)
            ids.append(id)

        return objects, labels, ids

    return None, None, None

def split_dataset(x, y, testRatio = 0.2, seed = 0):
    randomInstance = random.Random(seed)
    data = zip(x, y)
    trainSize = int((1.0 - testRatio)*len(y))
    randomInstance.shuffle(data)
    xTrain, yTrain = zip(*data[:trainSize])
    xTest, yTest = zip(*data[trainSize:])
    return xTrain, yTrain, xTest, yTest
