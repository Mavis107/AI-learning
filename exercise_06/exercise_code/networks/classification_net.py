from math import tanh
import numpy as np
import os
import pickle

from exercise_code.networks.layer import affine_forward, affine_backward, Sigmoid, Tanh, LeakyRelu, Relu
from exercise_code.networks.base_networks import Network


class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid, num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super().__init__("cifar10_classification_net")

        self.activation = activation()
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.std = std
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reset_weights()

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        self.eval()
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        self.eval()
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc
    
    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
        
        # Delete unnecessary caches, to mitigate a memory prolbem.
        self.reg = {}
        self.cache = {}
        
    def reset_weights(self):
        self.params = {'W1':self.std * np.random.randn(self.input_size, self.hidden_size),
                       'b1': np.zeros(self.hidden_size)}

        for i in range(self.num_layer - 2):
            self.params['W' + str(i + 2)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(self.hidden_size)

        self.params['W' + str(self.num_layer)] = self.std * np.random.randn(self.hidden_size,
                                                                  self.num_classes)
        self.params['b' + str(self.num_layer)] = np.zeros(self.num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(self.num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0
        

class MyOwnNetwork(ClassificationNet):
    """
    Custom neural network with modular architecture.
    """

    def __init__(self, activation=Relu, num_layer=2, 
                 input_size=3 * 32 * 32, hidden_size=100, 
                 std=1e-3, num_classes=10, reg=0, dropout_prob=0.2, use_batchnorm=True, use_dropout=True, **kwargs):
        """
        Initialize the network with weights, biases, and optional parameters.

        :param activation: Activation function (e.g., Relu, Sigmoid).
        :param num_layer: Number of layers in the network.
        :param input_size: Input dimension (e.g., 3*32*32 for CIFAR-10).
        :param hidden_size: Number of neurons in hidden layers.
        :param std: Standard deviation for weight initialization.
        :param num_classes: Number of output classes.
        :param reg: L2 regularization strength.
        :param dropout_prob: Dropout probability (default is 0.5).
        """
        super().__init__()

        self.activation = activation()
        self.num_layer = num_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.std = std
        self.reg_strength = reg
        self.dropout_prob = dropout_prob

        self.use_batchnorm = use_batchnorm  # New flag for BN
        self.use_dropout = use_dropout  # New flag for Dropout

        self.cache = {}
        self.reg = {}
        self.grads = {}

        self.reset_weights()

    def batchnorm_forward(self, X, gamma, beta):
        """Perform forward pass of batch normalization."""
        out = gamma * X + beta
        cache = (X, gamma, beta)
        return out, cache

    def batchnorm_backward(self, dout, cache):
        """Perform backward pass of batch normalization."""
        X, gamma, beta = cache
        dgamma = np.sum(dout * X, axis=0)
        dbeta = np.sum(dout, axis=0)
        dX = dout * gamma
        return dX, dgamma, dbeta

    def dropout_forward(self, X, p):
        """Apply dropout during forward pass."""
        mask = (np.random.rand(*X.shape) > p) / (1 - p)
        out = X * mask
        cache = mask
        return out, cache

    def dropout_backward(self, dout, cache):
        """Backpropagate through dropout."""
        mask = cache
        dX = dout * mask
        return dX

    def forward(self, X):
        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)  # Flatten input

        # Forward pass through layers
        for i in range(self.num_layer - 1):
            W, b = self.params[f'W{i + 1}'], self.params[f'b{i + 1}']

            # Affine transformation
            X, cache_affine = affine_forward(X, W, b)
            self.cache[f"affine{i + 1}"] = cache_affine

            # Batch normalization (if enabled)
            if self.use_batchnorm:
                gamma, beta = self.params.get(f'gamma{i + 1}', 1), self.params.get(f'beta{i + 1}', 0)
                X, cache_bn = self.batchnorm_forward(X, gamma, beta)
                self.cache[f"batchnorm{i + 1}"] = cache_bn

            # Activation function
            X, cache_activation = self.activation.forward(X)
            self.cache[f"activation{i + 1}"] = cache_activation

            # Dropout (if enabled)
            if self.use_dropout and self.dropout_prob > 0:
                X, dropout_cache = self.dropout_forward(X, self.dropout_prob)
                self.cache[f"dropout{i + 1}"] = dropout_cache

            # L2 regularization
            self.reg[f'W{i + 1}'] = np.sum(W ** 2) * self.reg_strength

        # Last layer: no activation, just affine
        W, b = self.params[f'W{self.num_layer}'], self.params[f'b{self.num_layer}']
        out, cache_affine = affine_forward(X, W, b)
        self.cache[f"affine{self.num_layer}"] = cache_affine
        self.reg[f'W{self.num_layer}'] = np.sum(W ** 2) * self.reg_strength

        return out

    def backward(self, dy):
        """
        Perform the backward pass through the network.

        :param dy: Gradient of loss wrt output.
        :return: Gradients of weights and biases.
        """
        dout = dy

        # Backward pass for last layer (affine only)
        cache_affine = self.cache[f"affine{self.num_layer}"]
        dout, dW, db = affine_backward(dout, cache_affine)
        self.grads[f'W{self.num_layer}'] = dW + 2 * self.reg_strength * self.params[f'W{self.num_layer}']
        self.grads[f'b{self.num_layer}'] = db

        # Backprop through remaining layers
        for i in range(self.num_layer - 2, -1, -1):
        # Dropout backward
            if self.use_dropout and self.dropout_prob > 0:
                dout = self.dropout_backward(dout, self.cache[f"dropout{i + 1}"])

            # Activation backward
            dout = self.activation.backward(dout, self.cache[f"activation{i + 1}"])

            # BatchNorm backward (only if BatchNorm was applied)
            if self.use_batchnorm:
                dout, dgamma, dbeta = self.batchnorm_backward(dout, self.cache[f"batchnorm{i + 1}"])
                self.grads[f'gamma{i + 1}'] = dgamma
                self.grads[f'beta{i + 1}'] = dbeta

            # Affine backward
            cache_affine = self.cache[f"affine{i + 1}"]
            dout, dW, db = affine_backward(dout, cache_affine)
            self.grads[f'W{i + 1}'] = dW + 2 * self.reg_strength * self.params[f'W{i + 1}']
            self.grads[f'b{i + 1}'] = db

        return self.grads

    def reset_weights(self):
        """Initialize weights and biases for the network."""
        self.params = {
            'W1': self.std * np.random.randn(self.input_size, self.hidden_size),
            'b1': np.zeros(self.hidden_size)
        }

        for i in range(1, self.num_layer - 1):
            self.params[f'W{i + 1}'] = self.std * np.random.randn(self.hidden_size, self.hidden_size)
            self.params[f'b{i + 1}'] = np.zeros(self.hidden_size)

        self.params[f'W{self.num_layer}'] = self.std * np.random.randn(self.hidden_size, self.num_classes)
        self.params[f'b{self.num_layer}'] = np.zeros(self.num_classes)

        # Initialize batch norm parameters
        for i in range(1, self.num_layer):
            self.params[f'gamma{i}'] = np.ones(self.hidden_size)
            self.params[f'beta{i}'] = np.zeros(self.hidden_size)

        self.grads = {}
