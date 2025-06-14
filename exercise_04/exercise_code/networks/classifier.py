
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super().__init__("classifier")

        self.num_features = num_features 
        self.W = None
        self.cache = None
        
        self.initialize_weights()

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point. 
            Note that it is changed to N x (D + 1) to include the bias term.
        :return: Predicted logits for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
                 
        Note: This simple neural-network contains TWO consecutive layers: 
        A fully-connected layer and a sigmoid layer.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        
        # Affine transformation output = X * W
        output = np.dot(X, self.W)

        # Sigmoid activation
        sigmoid_output = self.sigmoid(output)

        # Save the cache for backward pass
        self.cache = X, sigmoid_output

        return sigmoid_output

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. It is as the same shape of the forward() output.
                     If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'. (dL/dw)
        
        Note: Pay attention to the order in which we calculate the derivatives. It is the opposite of the forward pass!
        """
        assert self.cache is not None, "Run a forward pass before the backward pass. Also, don't forget to store the relevat variables\
            such as in 'self.cache = (X, y, ...)"
        
        # Retrieve cached values
        X, sigmoid_output =  self.cache 
        # Sigmoid derivative
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)
        # Calculate dL/dz
        dL_dz = dout * sigmoid_derivative
        # Gradient with respect to W
        dW = np.dot(X.T, dL_dz)

        return dW # (dL/dW)

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function.

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = 1/ (1 + np.exp(-x))

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
