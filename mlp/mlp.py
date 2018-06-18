import numpy as np

def softmax(x):
    ex = np.exp(-x)
    return ex / np.sum(ex)

def relu(x):
    return x*(x > 0.)

class MLP:
    def __init__(self, lr, bs, momentum, verbose, max_iters, eps):
        self.layers = []
        self.labels_ = []
        self.lr = lr
        self.batch_size = bs
        self.momentum = momentum
        self.verbose = verbose
        self.max_iters = max_iters
        self.eps = eps

    def init_layers_(self, num_inputs, num_labels):
        np.random.seed(0)
        self.layers = []
        hidden_layer = {'w':np.random.rand(num_inputs, 10), 'a':True}
        self.layers.append(hidden_layer)
        output_layer = {'w':np.random.rand(10, num_labels), 'a':False}
        self.layers.append(output_layer)

    def forward_(self, x, train=False):
        signal = x
        for layer in self.layers:
            signal = np.matmul(np.transpose(layer['w']), signal)
            if layer['a']:
                signal = relu(signal)
            if train:
                layer['output'] = signal
        return signal

    def backward_(self, expected):
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            if i == len(self.layers) - 1:
                errors = expected - softmax(current_layer['output'])

                current_layer['delta'] = error * current_layer['output']
            else:
                pass
            #current_layer['delta'] = errors
        pass

    def update_weights_(self):
        pass

    def init_train_iter_(self):
        pass

    def fit(self, x, y):
        num_samples = len(x)
        assert num_samples > 0
        assert num_samples == len(y)

        num_inputs = len(x[0])
        assert num_inputs > 0
        self.labels_ = np.unique(y)
        num_labels = len(self.labels_)
        assert num_labels > 0
        x = np.array(x)
        y = np.array(y)

        self.init_layers_(num_inputs, num_labels)

        np.random.seed(1)
        for i in range(self.max_iters):
            batch_indices = np.random.random_integers(0, num_samples - 1, self.batch_size)
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]

            self.init_train_iter_()
            for j in range(self.batch_size):
                outputs = softmax(self.forward_(batch_x[j], train=True))
                idx = np.argmax(outputs)
                label = self.labels_[idx]
                expected = (self.labels_ == batch_y[j]).astype(np.int8)
                self.backward_(expected)
            self.update_weights_()

    def predict(self, x):
        predictions = np.zeros(len(x))
        for i in range(len(x)):
            logits = softmax(self.forward_(x[i]))
            idx = np.argmax(logits)
            predictions[i] = self.labels_[idx]

        return predictions

    def score(self, x, y):
        assert len(x) == len(y)
        y = np.array(y).reshape(-1)

        predictions = self.predict(x)
        num_correct = np.sum(predictions == y)

        return float(num_correct) / y.shape[0]
