
import numpy as np


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth):
        loss = self.forward(y_out, y_truth)
        grad = self.backward(y_out, y_truth)
        return (loss, grad)


class L1(Loss):

    def forward(self, y_out, y_truth, reduction='mean'):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss for each sample of your training set.
        """

        result = np.abs(y_out - y_truth)

        if reduction == 'mean':
            result = result.mean()
        elif reduction == 'sum':
            result = result.sum()
        elif reduction == 'none':
            pass
        else:
            raise NotImplementedError

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for
                  each sample of your training set.
        """

        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1

        return gradient


class MSE(Loss):

    def forward(self, y_out, y_truth, reduction='mean'):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss for each sample of your training set.
        """

        result = (y_out - y_truth)**2

        if reduction == 'mean':
            result = result.mean()
        elif reduction == 'sum':
            result = result.sum()
        elif reduction == 'none':
            pass
        else:
            raise NotImplementedError

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for
                  each sample of your training set.
        """

        gradient = 2 * (y_out - y_truth)

        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth, reduction='mean'):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss for each sample of your training set.
        """

        result = -y_truth * np.log(y_out) - (1 - y_truth) * np.log(1 - y_out)
        if reduction == 'mean':
            result = result.mean()
        elif reduction == 'sum':
            result = result.sum()
        elif reduction == 'none':
            pass
        else:
            raise NotImplementedError

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out for
                  each sample of your training set.
        """

        gradient = - (y_truth / y_out) + (1 - y_truth) / (1 - y_out)

        return gradient
    
    
class CrossEntropyFromLogits(Loss):
    def __init__(self):
        self.cache = {}
     
    def forward(self, y_out, y_truth, reduction='mean'):
        """
        Performs the forward pass of the cross entropy loss function.
        
        :param y_out: [N, C] array with the predicted logits of the model
            (i.e. the value before applying any activation)
        :param y_truth: (N,) array with ground truth labels.
        :return: float, the cross-entropy loss value
        """
        
        # Prevent numerical overflow
        max_logits = np.max(y_out, axis=1, keepdims=True)    
        stabilized_logits = y_out - max_logits
        # Compute the logits' softmax probabilities
        exp_logits = np.exp(stabilized_logits)
        softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Compute the negative log likelihood for the correct class
        log_probs = -np.log(softmax_probs[np.arange(y_out.shape[0]), y_truth])

        if reduction == 'mean':
            loss = np.mean(log_probs)
        elif reduction == 'sum':
            loss = np.sum(log_probs)
        elif reduction == 'none':
            pass
        else:
            raise NotImplementedError

        # Store softmax probabilities for backward pass
        self.cache['softmax_probs'] = softmax_probs
        self.cache['y_truth'] = y_truth
      

        return loss
    
    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, C] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, C] array of cross entropy loss gradients w.r.t y_out for
                  each sample of your training set.
        """

        # Retrieve cached softmax probabilities
        softmax_probs = self.cache['softmax_probs']
        y_truth = self.cache['y_truth']
        # Initialize gradient 
        gradient = softmax_probs
        # Subtract 1 for the true class probabilities
        gradient[np.arange(y_out.shape[0]),y_truth] -= 1
        # Normalize by batch size
        gradient /= y_out.shape[0]
   
        return gradient
        
