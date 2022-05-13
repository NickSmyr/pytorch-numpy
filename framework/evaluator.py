# Want to be able to statically evaluate a model (for test set)
# Want to be able to attach to training process
#   And generate a list of metric values (loss, cost, accuracy)
from abc import ABC, abstractmethod

import numpy as np

from framework.model import Model, twolayer, ReLuDropoutBatchNormMLP

# TODO Share model forward pass on these
"""
An Evaluator evaluates outputs for each item in a batch by also
being able to extract information about a model along with the outputs

An Evaluator can use a precomputed output from the forward (
not-backward) pass of the model.
"""


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model: Model, X, Y, Y_pred):
        """
        Returns the evaluation of the model for inputs X,
        model outputs Y_pred, for input X, true outputs Y
        """
        pass


class AccuracyEvaluator(Evaluator):
    def evaluate(self, model: Model, X, Y, Y_pred):
        nd = Y.shape[1]
        Y = Y.argmax(axis=0)
        Y_pred = Y_pred.argmax(axis=0)
        return np.count_nonzero(Y_pred == Y) / nd


class twolayerCostEvaluator(Evaluator):
    def __init__(self, use_batchnorm):
        self.use_batchnorm = use_batchnorm

    def evaluate(self, model: twolayer, X, Y, Y_pred):
        nd = Y.shape[1]
        K = Y.shape[0]
        P = Y_pred
        # Get info from model
        lamda = model.lamda
        if isinstance(model, twolayer):
            regularizable_params = [model.params[0], model.params[2]]
        else:
            linear_w_period = 4 if self.use_batchnorm else 2
            regularizable_params = [x for i, x in enumerate(model.params) if
                                    i % linear_w_period == 0]

        cost = - (np.sum(Y * np.log(P)) / nd)
        for param in regularizable_params:
            cost += lamda * (np.sum(param ** 2))
        return cost


class twolayerLossEvaluator(Evaluator):
    def evaluate(self, model: twolayer, X, Y, Y_pred):
        nd = Y.shape[1]
        P = Y_pred
        cost = - (np.sum(Y * np.log(P)) / nd)
        return cost
