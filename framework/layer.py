## A layer has one or more inputs and a number of parameters
## ON forward pass one or more inputs are transformed to 1 output
## on backward pass we receive the gradient tensor of the output
##  we calculate the gradients of the local parameters
##  and we tranform the input gradients to output gradients

## For the assignment we will restrict to only 1 input tensor 1 input
## output layers in order to avoid creating a tensor graph
from abc import ABC, abstractmethod

## TODO we need to be able to have readable names for the parameters
## TODO (NOT URGENT) We need to implement the tensor graph for LSTMs
## TODO remove l2 from linear layer definition
from collections import defaultdict
from typing import List

import numpy as np

from framework.pure import relu, generate_dropout_mask, tanh
from test.convenience import zeros_like_list, np_copy_list, \
    hash_np_array


class Layer(ABC):
    def __init__(self, params, param_names=None, layer_name=None):
        self._params = params
        self._grads = zeros_like_list(self._params)
        self._param_names = param_names
        self._layer_name = layer_name

        self._test = False

        self.backward_values = defaultdict(lambda: [])

    def store_for_backward(self, name, value):
        """
        Stores the given value for the backward pass

        Should be called in each forward pass. A stack of values is built
        and when recover_backward_value is called it pops the stack of values

        :param name: the name of the value
        :param value: The value
        """
        self.backward_values[name].append(value)

    def recover_backward_value(self, name):
        """
        Recovers the last backward value from the forward pass. Also removes
        the value from the stack of stored values in forward

        :param name: the name of the value

        :return: the value
        """
        return self.backward_values[name].pop()

    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def backward(self, input_gradients):
        pass

    @property
    def grads(self):
        return np_copy_list(self._grads)

    @property
    def params(self):
        return np_copy_list(self._params)

    @property
    def param_names(self):
        if self._param_names is None:
            return ["unspec-param" for x in self._params]
        else:
            return self._param_names

    @property
    def layer_name(self):
        if self._layer_name is None:
            return "unspec-layer"
        else:
            return self._layer_name

    def set_test(self):
        """
        Sets the network to test mode
        """
        self._test = True

    def set_training(self):
        """
        Sets the network to train mode
        """
        self._test = False

    def is_test(self):
        """
        Returns true if layer is in test mode
        """
        return self._test == True

    def is_training(self):
        """
        Returns true if layer is in training mode
        """
        return self._test == False

    def set_params(self, params):
        self._params = params

    def zero_grads(self):
        self._grads = zeros_like_list(self._grads)


class Linear(Layer):

    def __init__(self, input_dim, output_dim, l2_lamda, initialization="he",
                 sigma=1e-3):
        """
        :param initialization: one of "he" or "normal". If normal each W
        param will follow a normal distribution with mean 0 and sigma
        """
        # Weight and bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma

        if initialization == "he":
            params = np.random.randn(output_dim, input_dim) / \
                     np.sqrt(input_dim), \
                     np.random.randn(output_dim, 1) * 0
        elif initialization == "normal":
            params = np.random.randn(output_dim, input_dim) * sigma, \
                     np.random.randn(output_dim, 1) * 0
        else:
            raise Exception("Initialization scheme must be either he or "
                            "normal")

        self.l2_lamda = l2_lamda
        super(Linear, self).__init__(params, ["W", "b"], "Linear")

    def __repr__(self):
        return f"(Linear ({self.input_dim},{self.output_dim}))"

    def forward(self, input_tensor):
        self.store_for_backward("input", input_tensor)
        output = self.params[0] @ input_tensor + self._params[1]
        # Save last output for backward pass
        return output

    def backward(self, input_gradients):
        # Calculate local gradients
        # Backpropagate gradients to lower layers
        last_input = self.recover_backward_value("input")
        nd = last_input.shape[-1]
        gradW = ((input_gradients @ last_input.T) / nd) + 2 * \
                self.l2_lamda * self.params[0]
        gradb = np.sum(input_gradients, axis=1, keepdims=True) / nd
        self._grads = [gradW + self._grads[0], gradb + self._grads[1]]
        return self.params[0].T @ input_gradients

class ReLu(Layer):
    def __init__(self):
        super(ReLu, self).__init__([], [], "relu")

    def forward(self, input_tensor):
        output = relu(input_tensor)
        self.store_for_backward("output", output)
        return output

    def backward(self, input_gradients):
        last_output = self.recover_backward_value("output")
        output_gradients = np.copy(input_gradients)
        output_gradients[last_output <= 0] = 0
        return output_gradients

class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__([], [], "tanh")

    def forward(self, input_tensor):
        self.store_for_backward("input", input_tensor)
        output = tanh(input_tensor)
        return output

    def backward(self, input_gradients):
        last_input = self.recover_backward_value("input")
        output_gradies = input_gradients * ( 1 - (tanh(last_input) ** 2))
        return output_gradies



class Dropout(Layer):
    def __init__(self, pdrop):
        super().__init__([], [], "dropout")
        self.pdrop = pdrop

    def forward(self, input_tensor):
        if self.is_test():
            return input_tensor
        dropout = generate_dropout_mask(self.pdrop,
                                                  input_tensor.shape)
        self.store_for_backward("dropout", dropout)
        self.last_output = (input_tensor * dropout) / (1 -
                                                                 self.pdrop)
        return self.last_output

    def backward(self, input_gradients):
        last_dropout = self.recover_backward_value("dropout")
        return (input_gradients * last_dropout) / \
               (1 - self.pdrop)


class Sequential(Layer):
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        params = []
        param_names = []
        for i, l in enumerate(layers):
            params.extend(l.params)
            param_names.extend([f"{l.layer_name}-{i}-{x}" for x in
                                l.param_names])

        super(Sequential, self).__init__(params, param_names, "sequential")

    def forward(self, input_tensor):
        curr_tensor = input_tensor
        for layer in self.layers:
            curr_tensor = layer.forward(curr_tensor)
        return curr_tensor

    # Overrides
    def backward(self, input_gradients):
        curr_grads = input_gradients
        for i, layer in enumerate(reversed(self.layers)):
            curr_grads = layer.backward(curr_grads)
            pass
        # TODO we should not need to do this
        # At the end of the loop we need to set the gradients for
        # The seq model
        self._grads = []
        for layer in self.layers:
            self._grads.extend(layer.grads)
        return curr_grads

    @property
    def grads(self):
        grad_list = []
        for layer in self.layers:
            grad_list.extend(layer.grads)
        return grad_list

    ## TODO we should not need to do this
    def set_params(self, params):
        """This has to be overriden because the nested layers must
        have their layers updated"""
        # Super call
        super(Sequential, self).set_params(params)

        curr_i = 0
        for x in self.layers:
            x_params = x.params
            x_params_length = len(x_params)
            end_i = curr_i + x_params_length
            x.set_params(params[curr_i: end_i])
            curr_i = end_i

    def set_test(self):
        super(Sequential, self).set_test()
        for layer in self.layers:
            layer.set_test()

    def set_training(self):
        super(Sequential, self).set_training()
        for layer in self.layers:
            layer.set_training()

    def zero_grads(self):
        super(Sequential, self).zero_grads()
        for layer in self.layers:
            layer.zero_grads()




class BatchNormMeanVarianceStrategy(ABC):
    """
    A batch norm mean variance strategy selects the mean and variance to use
    in the forward pass based on the layers state
    """

    @abstractmethod
    def select_mean_variance(self, bn_layer: "BatchNorm", batch_mean,
                             batch_variance):
        """
        Select mean and variance to use, called whenever forward is called
        on the bn_layer

        :param bn_layer: The BatchNorm layer
        :param batch_mean: The batch mean of each batch
        :param batch_variance: The batch variance of each batch
        """


class FixedMeanVarianceOnTestBatchNormStrategy(BatchNormMeanVarianceStrategy):
    """
    Use only the given mean and variance whenever the layer is on test mode,
    otherwise use the given batch_mean or batch_variance
    """

    def __init__(self, mean_to_use, var_to_use):
        self.mean_to_use = mean_to_use
        self.var_to_use = var_to_use

    def select_mean_variance(self, bn_layer: "BatchNorm", batch_mean,
                             batch_variance):
        if bn_layer.is_test():
            return self.mean_to_use, self.var_to_use
        else:
            return batch_mean, batch_variance

class FixedBNStrategy(BatchNormMeanVarianceStrategy):
    def select_mean_variance(self, bn_layer: "BatchNorm", batch_mean,
                             batch_variance):
        return batch_mean, batch_variance


class DefaultBNStrategy(BatchNormMeanVarianceStrategy):

    def __init__(self, alpha):
        self.alpha = alpha

        self.last_mean = None
        self.last_variance = None

    def select_mean_variance(self, bn_layer: "BatchNorm", batch_mean,
                             batch_variance):
        """
        Calculate the mean and variance that will be used for scaling the input
        datapoints. Inputs are the batch_mean and batch_variance calculated
        on every forward call
        """
        # If at test time update test mean and variance
        if bn_layer.is_test():
            # If this is the first minibatch of the test set, set the given
            # means to be the means of the first batch
            if bn_layer.on_first_forward_call_on_test:
                mean = batch_mean
                variance = batch_variance
            else:
                mean = self.alpha * self.last_mean + (
                        1 - self.alpha) * \
                       batch_mean
                variance = self.alpha * self.last_variance \
                           + (1 - self.alpha) * batch_variance

        # If used during training then just use the batch mean and variance
        else:
            mean = batch_mean
            variance = batch_variance

        self.last_mean = mean
        self.last_variance = variance

        return mean, variance


class BatchNorm(Layer):
    def __init__(self, input_dim, alpha, strategy :
    BatchNormMeanVarianceStrategy=None):
        self.input_dim = input_dim
        self.eps = np.finfo(float).eps
        self.alpha = alpha

        self.on_first_forward_call_on_test = False

        gamma = np.ones((input_dim, 1))
        beta = np.zeros((input_dim, 1))

        if strategy==None:
            self.strategy = DefaultBNStrategy(alpha)

        super(BatchNorm, self).__init__([gamma, beta],
                                        ["gamma", "beta"],
                                        "batchnorm")

    def set_strategy(self, strategy : BatchNormMeanVarianceStrategy):
        self.strategy = strategy

    def set_test(self):
        super(BatchNorm, self).set_test()
        self.on_first_forward_call_on_test = True

    def set_training(self):
        super(BatchNorm, self).set_training()
        self.on_first_forward_call_on_test = False

    def forward(self, input_tensor):
        # Input sensor input_dim x n_batch
        n_batch = input_tensor.shape[1]
        self.last_input_tensor = input_tensor

        new_mean = np.sum(input_tensor, axis=1, keepdims=True) / n_batch
        # input_dim x 1
        new_variance = np.sum((input_tensor - new_mean) ** 2,
                              axis=1, keepdims=True) / n_batch

        self.last_mean, self.last_variance = \
            self.strategy.select_mean_variance(self, new_mean, new_variance)

        self.last_normalized = (input_tensor - self.last_mean) / np.sqrt(
            self.last_variance + self.eps)

        # Rescale
        self.last_rescaled = self.last_normalized * self.params[0] + \
                             self.params[1]

        # Unset first test forward call flag
        self.on_first_forward_call_on_test = False
        return self.last_rescaled

    def backward(self, input_gradients):
        n_batch = input_gradients.shape[1]

        gamma_grad = np.sum(input_gradients * self.last_normalized, axis=1,
                            keepdims=True) / n_batch
        beta_grad = np.sum(input_gradients, axis=1, keepdims=True) / n_batch

        self._grads[0] = gamma_grad
        self._grads[1] = beta_grad

        # Propagate gradients
        output_gradients = input_gradients * self.params[0]
        sigma1 = (self.last_variance + self.eps) ** (-0.5)
        sigma2 = (self.last_variance + self.eps) ** (-1.5)

        g1 = output_gradients * sigma1  # input_dim x n_batch
        g2 = output_gradients * sigma2  # input_dim x n_batch

        d = (self.last_input_tensor - self.last_mean)  # input_dim x n_batch
        c = np.sum(g2 * d, axis=1, keepdims=True)  # input_dim x n_batch
        output_gradients = g1 - (np.sum(g1, axis=1, keepdims=True) / n_batch) \
                           - ((d * c) / n_batch)
        return output_gradients


class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__([], [], "identity")

    def forward(self, input_tensor):
        return input_tensor

    def backward(self, input_gradients):
        return input_gradients

class Residual(Layer):
    def __init__(self, underlying_layer : Layer):
        self.layer = underlying_layer
        super(Residual, self).__init__(underlying_layer.params,
                                       underlying_layer.param_names, "residual")



    def forward(self, input_tensor):
        return self.layer.forward(input_tensor) + input_tensor

    def backward(self, input_gradients):
        layer_output_gradients  = self.layer.backward(input_gradients)
        return layer_output_gradients + input_gradients

    def set_params(self, params):
        self.layer.set_params(params)

    @property
    def grads(self):
        return self.layer.grads

    @property
    def params(self):
        return self.layer.params

    @property
    def param_names(self):
        return self.layer.param_names

    @property
    def layer_name(self):
        return self.layer.layer_name

    def set_test(self):
        self.layer.set_test()

    def set_training(self):
        self.layer.set_training()

    def is_test(self):
        return self.layer.is_test()

    def is_training(self):
        return self.layer.is_training()

    def zero_grads(self):
        self.layer.zero_grads()

class OneHotEmbeddingLayer(Layer):
    def __init__(self, set_of_elements : set):
        """
        :param set_of_elements: The set of elements to generate embeddings
        :param emb_size: The size of the embeddings
        """
        self.elem_list = list(set_of_elements)
        self.elem2i = {v :i  for i,v in enumerate(self.elem_list)}
        self.i2elem = self.elem_list

        #embedding_matrix = np.random.randn(len(self.elem_list), emb_size)
        embedding_matrix = np.eye(len(self.elem_list))

        super(OneHotEmbeddingLayer, self).__init__([embedding_matrix],
                                                   ["embedding_mat"],
                                             "Embedding")

    def forward(self, input_tensor):
        """
        :param input_tensor: Nd array of shape (num_elements,
        seq_length, n_batch).
        Each
        element in input array must be within the given set_of_elements that
        is set during initialization :return: nd array of shape (emb_size,
        n_batch)
        """
        def numerify(x):
            return self.elem2i[x]

        self.last_input = np.vectorize(numerify)(input_tensor)
        self.last_output = self.params[0][self.last_input.T].T
        return self.last_output

    def decode(self, input_tensor):
        """
        Converts a tensor of indexes to the underlying elements
        :param input_tensor: nd array of shape (seq_length, n_batch)
        """
        def denumerify(x):
            return self.i2elem[x]

        output = np.vectorize(denumerify)(input_tensor) # seq_length x n_batch
        return output

    def backward(self, input_gradients):
        raise NotImplementedError
