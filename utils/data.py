import random
import numpy as np
import csv

def load_data(data_path):
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)
        train = 'y' in header
        objects = []
        labels = []
        ids = []
        for row in reader:
            if train:
                x = [float(row[i]) for i in range(1, len(row) - 1)]
                y = int(row[-1])
                labels.append(y)
            else:
                x = [float(row[i]) for i in range(1, len(row))]
            id = int(float(row[0]))
            objects.append(x)
            ids.append(id)

        return objects, labels, ids

    return None, None, None

def save_predictions(filename, ids, predictions):
    assert len(ids) == len(predictions)

    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['ID', 'y'])
        for i, id in enumerate(ids):
            writer.writerow([id, predictions[i]])

def split_dataset(x, y, testRatio = 0.2, seed = 0):
    randomInstance = random.Random(seed)
    data = list(zip(x, y))
    trainSize = int((1.0 - testRatio)*len(y))
    randomInstance.shuffle(data)
    xTrain, yTrain = zip(*data[:trainSize])
    xTest, yTest = zip(*data[trainSize:])
    return xTrain, yTrain, xTest, yTest

class DataNormalizer:
    def __init__(self):
        self.mean_ = []
        self.std_ = []

    def fit(self, objects):
        tmp_data = np.matrix(objects)
        self.mean_ = np.array(np.mean(tmp_data, axis=0)).reshape(-1)
        self.std_ = np.array(np.std(tmp_data, axis=0)).reshape(-1)

    def transform(self, objects):
        return (objects - self.mean_) / self.std_
