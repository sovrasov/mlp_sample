import numpy as np

class MLP:
    def __init__(self):
        self.labels_ = []

    def fit(self, x, y):
        self.labels_ = np.unique(y)
        pass

    def predict(self, x):
        return np.array([self.labels_[0]]*len(x))

    def score(self, x, y):
        assert len(x) == len(y)
        y = np.array(y).reshape(-1)

        predictions = self.predict(x)
        num_correct = np.sum(predictions == y)

        return float(num_correct) / y.shape[0]
