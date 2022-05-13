from typing import List

import numpy as np

from test.convenience import np_copy_list, hash_np_array, zeros_like_list
from .augmentation import Assignment2Augmenter
from .layer import Linear, ReLu, Dropout, Sequential, BatchNorm, Identity, \
    FixedMeanVarianceOnTestBatchNormStrategy, FixedBNStrategy, Residual, Tanh, \
    OneHotEmbeddingLayer
from .pure import relu, generate_dropout_mask, softmax
from abc import ABC, abstractmethod

"""
A model should hold all the tunable parts -> Also optimizer
"""


# TODO Separate assignment 2 logic from abstract framework logic
# TODO Refactor
class Model(ABC):
    @property
    @abstractmethod
    def grads(self):
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def zero_grads(self):
        pass

    @abstractmethod
    def forward(self, X=None, Y=None, do_backward=False):
        pass

    @abstractmethod
    def set_params(self, params: List[np.ndarray]):
        pass

    def set_test(self):
        self._test_mode = False

    def set_training(self):
        self.test_mode = False

    def param_names(self):
        pass


class twolayer(Model):
    def __init__(self, K, d, m,
                 lamda, pdrop, pflip, ptrans,
                 num_channels=3, image_width=32, image_height=32):
        """
        d=input dim
        m=hidden dim
        """
        super(twolayer, self).__init__()

        self.lamda = lamda
        self.pdrop = pdrop
        self.augmenter = Assignment2Augmenter(pflip, ptrans,
                                              num_channels,
                                              image_width,
                                              image_height)
        self.K = K
        self.d = d
        self.m = m

        self._params = []

        W1, b1, W2, b2 = np.random.randn(m, d) / np.sqrt(d), \
                         np.random.randn(m, 1) * 0, \
                         np.random.randn(K, m) / np.sqrt(m), \
                         np.random.randn(K, 1) * 0

        self._params.extend([W1, b1, W2, b2])

        self._grads = [np.zeros_like(x) for x in self.params]

    def forward(self, X=None, Y=None, do_backward=False):
        """
        Forward_backward pass. If Y is provided will return the loss, otherwise will return
        probabilities. If Y is provided and do_backward is also provided the gradients are filled
        """
        nd = X.shape[1]
        # Generate dropout
        dropout = generate_dropout_mask(self.pdrop, (self.m, nd))

        # Apply data augmentation only if training
        if do_backward == True:
            X = self.augmenter.apply(X)

        # Forwards
        Hbatch = relu(self.params[0] @ X + self.params[1])
        Hdrop = (Hbatch * dropout) / (1 - self.pdrop)
        Pbatch = self.params[2] @ Hdrop + self.params[3]
        Pbatch = softmax(Pbatch)
        # Save for backward

        if do_backward and Y is not None:
            Gbatch = - (Y - Pbatch)
            grads = [None] * 4
            grads[2] = Gbatch @ Hdrop.T / nd + 2 * self.lamda * self.params[2]
            grads[3] = np.sum(Gbatch, axis=1, keepdims=True) / nd

            # "Propagating gradient"
            # Gbatch is dl/dHhidden_ij
            Gbatch = self.params[2].T @ Gbatch
            # Backprop through dropout layer
            Gbatch = (Gbatch * dropout) / (1 - self.pdrop)
            Gbatch[Hbatch <= 0] = 0
            grads[0] = Gbatch @ X.T / nd + 2 * self.lamda * self.params[0]
            grads[1] = np.sum(Gbatch, axis=1, keepdims=True) / nd

            self._grads = grads

        if Y is None:
            return Pbatch
        else:
            return - (np.sum(Y * np.log(Pbatch)) / nd) + \
                   (self.lamda * (np.sum(self.params[0] ** 2)
                                  + np.sum(self.params[2] ** 2)))

    def set_params(self, params):
        """
        Sets the parameters of the network
        """
        self._params = params

    @property
    def params(self):
        """
        Returns the params of the network
        """
        return np_copy_list(self._params)

    @property
    def grads(self):
        """
        Returns the gradients for the previous backward pass
        """
        return np_copy_list(self._grads)

    def zero_grads(self):
        """
        Zeros the gradients
        """
        self._grads = [np.zeros_like(x) for x in self.grads]


class ReLuDropoutBatchNormMLP(Model):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_sizes: List[int], lamda, pdrop, pflip,
                 ptrans, num_channels=3, image_width=32, use_batchnorm=False,
                 batch_norm_alpha=0.9,
                 image_height=32,
                 initialization="he",
                 batch_norm_after_activation=False,
                 sigma=1e-4):
        """
        MLP with blocks consting of a linear layer, a ReLu
        activation and a dropout layer
        """
        super(Model, self).__init__()

        self.layer_sizes = [input_dim] + hidden_sizes
        self.use_batchnorm = use_batchnorm

        # Add blocks to layers
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            # Extend with intermediate block
            block = [
                Linear(self.layer_sizes[i], self.layer_sizes[i + 1],
                       lamda, initialization=initialization, sigma=sigma)
            ]
            bn_layer = BatchNorm(self.layer_sizes[i + 1], batch_norm_alpha) \
                if use_batchnorm else Identity()
            if batch_norm_after_activation:
                block.extend([bn_layer, ReLu()])
            else:
                block.extend([ReLu(), bn_layer])
            block.append(Dropout(pdrop))
            self.layers.extend(block)

        # Add final block
        self.layers.append(
            Linear(self.layer_sizes[-1], output_dim, lamda)
        )
        # Create the sequential block
        self.seq = Sequential(self.layers)

        self.lamda = lamda
        self.pdrop = pdrop
        self.augmenter = Assignment2Augmenter(pflip, ptrans,
                                              num_channels,
                                              image_width,
                                              image_height)

    def set_test(self):
        super(ReLuDropoutBatchNormMLP, self).set_test()
        self.seq.set_test()

    def set_training(self):
        super(ReLuDropoutBatchNormMLP, self).set_training()
        self.seq.set_training()

    def forward(self, X=None, Y=None, do_backward=False):
        """
        Forward_backward pass. If Y is provided will return the
        loss, otherwise will return probabilities. If Y is provided
        and do_backward is also provided the gradients are filled
        """
        nd = X.shape[1]
        # Apply data augmentation only if training
        if do_backward:
            X = self.augmenter.apply(X)

        Pbatch = self.seq.forward(X)
        Pbatch = softmax(Pbatch)
        # Save for backward

        if do_backward and Y is not None:
            Gbatch = - (Y - Pbatch)
            self.seq.backward(Gbatch)

        if Y is None:
            return Pbatch
        else:
            # TODO calculate loss with l2 using graph
            loss = - (np.sum(Y * np.log(Pbatch)) / nd)
            for i, param in enumerate(self.params):
                ## Only add regularization for the weight parameters
                ## of the linear layer
                w_layer_period = 4 if self.use_batchnorm else 2
                if i % w_layer_period == 0:
                    loss += self.lamda * np.sum(param ** 2)
            return loss

    def fix_bn_means_and_variances(self, X):
        """
        Fix means and variances for the batch norm layers to be the output
        mean and variances of the BatchNorm layers for given batch of inputs X
        """
        if not self.use_batchnorm:
            raise Exception("This model must use batch norm in order to "
                            "estimate means and variances")
        # Before forward disable expma
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.set_strategy(FixedBNStrategy())

        model_output = self.forward(X, do_backward=False)
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                m, v = layer.last_mean, layer.last_variance
                layer.set_strategy(FixedMeanVarianceOnTestBatchNormStrategy(
                    m, v))

    def set_params(self, params):
        """
        Sets the parameters of the network
        """
        self.seq.set_params(params)

    @property
    def params(self):
        """
        Returns the params of the network
        """
        return self.seq.params

    @property
    def grads(self):
        """
        Returns the gradients for the previous backward pass
        """
        return self.seq.grads

    def zero_grads(self):
        """
        Zeros the gradients
        """
        self.seq.zero_grads()

    def param_names(self):
        return self.seq.param_names


class SkipMLP(Model):
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int, num_layers: int, lamda, pdrop, pflip,
                 ptrans, num_channels=3, image_width=32, use_batchnorm=False,
                 batch_norm_alpha=0.9,
                 image_height=32,
                 initialization="he",
                 batch_norm_after_activation=False,
                 sigma=1e-4):
        """
        MLP with blocks consting of a linear layer, a ReLu
        activation and a dropout layer
        """
        super(Model, self).__init__()
        self.use_batchnorm = use_batchnorm

        # Add blocks to layers
        self.layers = []
        # Add initial layer
        self.layers.append(Linear(input_dim, hidden_dim, lamda,
                                  initialization=initialization, sigma=sigma))

        for i in range(num_layers):
            # Extend with intermediate block
            block = []

            residual_block = Sequential([
                Linear(hidden_dim, hidden_dim, lamda,
                       initialization=initialization, sigma=sigma),
                ReLu(),
                Linear(hidden_dim, hidden_dim, lamda,
                       initialization=initialization,
                       sigma=sigma)
            ])
            residual_block = Residual(residual_block)

            # Add residual to block
            block.append(residual_block)
            # Add the rest
            bn_layer = BatchNorm(hidden_dim, batch_norm_alpha) \
                if use_batchnorm else Identity()
            if batch_norm_after_activation:
                block.extend([bn_layer, ReLu()])
            else:
                block.extend([ReLu(), bn_layer])
            block.append(Dropout(pdrop))
            self.layers.extend(block)

        # Add final block
        self.layers.append(
            Linear(hidden_dim, output_dim, lamda,
                   initialization=initialization,
                   sigma=sigma)
        )
        # Create the sequential block
        self.seq = Sequential(self.layers)

        self.lamda = lamda
        self.pdrop = pdrop
        self.augmenter = Assignment2Augmenter(pflip, ptrans,
                                              num_channels,
                                              image_width,
                                              image_height)

    def set_test(self):
        super(SkipMLP, self).set_test()
        self.seq.set_test()

    def set_training(self):
        super(SkipMLP, self).set_training()
        self.seq.set_training()

    def forward(self, X=None, Y=None, do_backward=False):
        """
        Forward_backward pass. If Y is provided will return the
        loss, otherwise will return probabilities. If Y is provided
        and do_backward is also provided the gradients are filled
        """
        nd = X.shape[1]
        # Apply data augmentation only if training
        if do_backward:
            X = self.augmenter.apply(X)

        Pbatch = self.seq.forward(X)
        Pbatch = softmax(Pbatch)
        # Save for backward

        if do_backward and Y is not None:
            Gbatch = - (Y - Pbatch)
            self.seq.backward(Gbatch)

        if Y is None:
            return Pbatch
        else:
            loss = - (np.sum(Y * np.log(Pbatch)) / nd)
            for i, param in enumerate(self.params):
                ## Only add regularization for the weight parameters
                ## of the linear layer
                w_layer_period = 4 if self.use_batchnorm else 2
                if i % w_layer_period == 0:
                    loss += self.lamda * np.sum(param ** 2)
            return loss

    def fix_bn_means_and_variances(self, X):
        """
        Fix means and variances for the batch norm layers to be the output
        mean and variances of the BatchNorm layers for given batch of inputs X
        """
        if not self.use_batchnorm:
            raise Exception("This model must use batch norm in order to "
                            "estimate means and variances")
        # Before forward disable expma
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                layer.set_strategy(FixedBNStrategy())

        model_output = self.forward(X, do_backward=False)
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                m, v = layer.last_mean, layer.last_variance
                layer.set_strategy(FixedMeanVarianceOnTestBatchNormStrategy(
                    m, v))

    def set_params(self, params):
        """
        Sets the parameters of the network
        """
        self.seq.set_params(params)

    @property
    def params(self):
        """
        Returns the params of the network
        """
        return self.seq.params

    @property
    def grads(self):
        """
        Returns the gradients for the previous backward pass
        """
        return self.seq.grads

    def zero_grads(self):
        """
        Zeros the gradients
        """
        self.seq.zero_grads()

    def param_names(self):
        return self.seq.param_names


class VanillaRNN(Model):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_steps, element_set, sampling_strat : str = "normal",
                 temperature = 1, theta=1):
        super(VanillaRNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = OneHotEmbeddingLayer(element_set)

        ## TODO disable bias on one of these
        self.wlinear = Linear(hidden_dim, hidden_dim, 0,
                              initialization="normal",
                              sigma=0.01)
        self.ulinear = Linear(input_dim, hidden_dim, 0,
                              initialization="normal",
                              sigma=0.01)

        self.tanh_layer = Tanh()
        self.output_linear = Linear(hidden_dim, output_dim, 0,
                                    initialization="normal", sigma=0.01)

        self.num_steps = num_steps

        ## Only need this to group parametable layers
        self._seq = Sequential([
            self.wlinear,
            self.ulinear,
            self.output_linear
        ])

        self._hidden_for_next_forward = np.zeros((hidden_dim, 1))

        self.sampling_strat = sampling_strat
        self.temperature = temperature
        self.theta = theta

    def set_sampling_strat(self, strat):
        self.sampling_strat = strat

    def set_hidden_for_next_formard(self, hidden):
        """
        Sets the initial hidden states for the next forward.
        :param hidden: ndarray of shape (hidden_dim , n_batch)
        """
        self._hidden_for_next_forward = hidden

    def reset_hidden_state(self):
        self._hidden_for_next_forward = np.zeros((self.hidden_dim, 1))

    @property
    def grads(self):
        return self._seq.grads

    @property
    def params(self):
        return self._seq.params

    def param_names(self):
        return self._seq.param_names

    def zero_grads(self):
        self._seq.zero_grads()

    def generate_sequence(self, initial_char : str, seq_length : int):
        """
        Returns a string sequence of the model
        """
        # Make X shape (1,1,1)
        X = np.array(initial_char)[np.newaxis, ... , np.newaxis]
        for i in range(seq_length):
            Y_pred = self.forward(X, num_steps=i+1) # Shape (seq, 1)
            # Add last prediction to input
            X = np.concatenate([X, Y_pred[-1:,:]], axis=0)

        # Now X is of shape (1, seq+1, 1)
        return "".join(X[1:,0])

    def sample_from_distr(self, distr):
        """
        :param distr: An array of positive reals summing to one, the input
        distribution to sample from
        :return : an int representing the sample
        """

        print("Tets pls work")
        if self.sampling_strat == "nucleus":
            idxs = list(range(len(distr)))
            idxs = sorted(idxs , key=lambda x : distr[x])
            distr = sorted(distr)
            curr_sum = 0
            # Get top K probabilities summing to at least theta
            k = len(distr)
            for i in range(len(distr)):
                curr_sum += distr[i]
                if curr_sum > self.theta:
                    k = i
                    break
            idxs = idxs[k:]
            distr = np.array(distr[k:])
            # normalize distr
            distr = distr / np.sum(distr)
            return np.random.choice(idxs, p=distr)
        else:
            return np.random.choice(list(range(len(distr))), p=distr)

    def sample_from_output_array(self, Ypred):
        """
        :param Ypred: The output probabilities of shape (output_dim, seq, bs)
        :return: array (seq,bs) of output idxs
        """
        output_arr = np.zeros((Ypred.shape[1], Ypred.shape[2]), dtype=np.int)
        for bs in range(0, Ypred.shape[2]):
            for seq in range(0, Ypred.shape[1]):
                distr = Ypred[:, seq, bs]
                output = self.sample_from_distr(distr)
                output_arr[seq, bs] = output

        return output_arr

    def forward(self, X=None, Y=None, do_backward=False, num_steps=None):
        """
        :param X: must be a tensor of shape (seq_size, batch_size)
        """
        batch_size = X.shape[1]
        seq_size = X.shape[0]

        # Batch initial hidden states
        if len(self._hidden_for_next_forward.shape)  == 1:
            self._hidden_for_next_forward = self._hidden_for_next_forward[\
                                                ...,np.newaxis]

        # Broadcast to (hidden_dim, bs) if not already
        if self._hidden_for_next_forward.shape[1] == 1:
            self._hidden_for_next_forward = np.broadcast_to(
                self._hidden_for_next_forward, (self.hidden_dim, batch_size))

        X = self.embedding.forward(X)
        hidden_states = [] # List of (hidden_dim, n_batch_vectors)
        predictions = []
        hidden_states.append(self._hidden_for_next_forward)

        # Need the option to set the number of steps in forward
        if num_steps is None:
            steps = seq_size
        else:
            steps = num_steps

        for i in range(steps):
            hidden_scores = self.wlinear.forward(hidden_states[-1]) + \
                            self.ulinear.forward(X[:,i,:])
            hidden_activations = self.tanh_layer.forward(hidden_scores)
            hidden_states.append(hidden_activations)

            output_logits = self.output_linear.forward(hidden_activations)
            if num_steps is not None and self.sampling_strat == "temperature":
                output_probabilities = softmax(output_logits,0,
                                               temperature=self.temperature)
            else:
                output_probabilities = softmax(output_logits,0)
            # output_probabilities shape (output_dim, n_batch)
            predictions.append(output_probabilities)

        # Shape output_dim x seq_len x n_batch
        Ypred = np.concatenate([np.expand_dims(x, axis=1) for x in \
                                          predictions], axis=1)

        self.forward_last_hidden = hidden_states[-1]
        if Y is not None:
            # Convert Y to one hot
            Y = self.embedding.forward(Y)
            # Compute loss
            loss = - np.sum((Y * np.log(Ypred))) / batch_size
            if do_backward:
                Gbatch = - (Y - Ypred) # out_dim x seq x bs
                previous_hidden_G = np.zeros((self.hidden_dim , 1))
                for i in range(steps):
                    input_G = Gbatch[:, steps - i - 1,:]
                    input_G = self.output_linear.backward(input_G)
                    input_G = self.tanh_layer.backward(input_G + previous_hidden_G)

                    _ = self.ulinear.backward(input_G)
                    # Backprop
                    previous_hidden_G = self.wlinear.backward(
                        input_G)
            return loss
        else:
            # Return sampeld outputs
            Ypred = self.sample_from_output_array(Ypred)
            Ypred = self.embedding.decode(Ypred)
            return Ypred

    def set_params(self, params: List[np.ndarray]):
        self._seq.set_params(params)

def print_param_grad_summary(model):
    print("========= PARAM + GRAD SUMMARY =========")
    print("param name | max param | min param | max grad | min grad ")
    for param, param_name, grad in zip(model.params,
                                       model.param_names(),
                                       model.grads):
        print(param_name, np.max(param), np.min(param), np.max(grad),
              np.min(grad))