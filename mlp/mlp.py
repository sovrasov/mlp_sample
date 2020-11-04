import numpy as np

def softmax(x):
    ex = np.exp(-x)
    return ex / np.sum(ex)

def relu(x):
    return x*(x > 0.)

def relu_der(x):
    return np.ones_like(x)* (x > 0.)

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

    def _create_layer(self, num_inputs, num_outputs, activate=True):
        return {'w':np.random.rand(num_inputs, num_outputs), 'b': np.random.rand(num_outputs), 'a':activate,
                'batch_grad_w':np.zeros((num_inputs, num_outputs), dtype=np.float32),
                'batch_grad_b':np.zeros(num_outputs, dtype=np.float32)}

    def init_layers_(self, num_inputs, num_labels):
        np.random.seed(0)
        self.layers = []
        hidden_layer = self._create_layer(num_inputs, 10, True)
        self.layers.append(hidden_layer)
        output_layer = self._create_layer(10, num_labels, False)
        self.layers.append(output_layer)

    def forward_(self, x, train=False):
        signal = x
        for layer in self.layers:
            if train: layer['input'] = np.copy(signal)
            signal = np.matmul(np.transpose(layer['w']), signal) + layer['b']
            if layer['a']:
                if train: layer['pre_output'] = signal
                signal = relu(signal)
        return signal

    def backward_(self, expected, outputs):
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            if i == len(self.layers) - 1: # handle the last layer
                errors = expected - outputs
                current_layer['delta'] = errors
                if current_layer['a']:
                    current_layer['delta'] *= relu_der(current_layer['pre_output'])
            else:
                next_layer = self.layers[i + 1]
                current_layer['delta'] = np.matmul(next_layer['w'], next_layer['delta']) * \
                                         relu_der(current_layer['pre_output'])
            current_layer['batch_grad_b'] += current_layer['delta']
            current_layer['batch_grad_w'] += np.matmul(current_layer['input'].reshape(-1, 1),
                                                       current_layer['delta'].reshape(1, -1))

    def update_weights_(self):
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            current_layer['batch_grad_b'] /= self.batch_size
            current_layer['batch_grad_w'] /= self.batch_size
            current_layer['b'] -= self.lr * current_layer['batch_grad_b']
            current_layer['w'] -= self.lr * current_layer['batch_grad_w']

    def init_train_iter_(self):
        for layer in self.layers:
            layer['batch_grad_b'] *= 0.
            layer['batch_grad_w'] *= 0.

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
                self.backward_(expected, outputs)
            self.update_weights_()

    def predict(self, x):
        predictions = np.zeros(len(x))
        for i in range(len(x)):
            probs = softmax(self.forward_(x[i]))
            idx = np.argmax(probs)
            predictions[i] = self.labels_[idx]

        return predictions

    def score(self, x, y):
        assert len(x) == len(y)
        y = np.array(y).reshape(-1)

        predictions = self.predict(x)
        num_correct = np.sum(predictions == y)

        return float(num_correct) / y.shape[0]
