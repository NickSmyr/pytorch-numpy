from typing import List

import numpy as np

from framework.model import Model
from test.convenience import generate_random_one_hot


class RandomModel(Model):
    def __init__(self, K):
        self._params = [1]
        self._grads = [0]

        self.K = K

    @property
    def grads(self):
        return self.grads

    @property
    def params(self):
        return self._params

    def zero_grads(self):
        self._grads = [0]

    def forward(self, X=None, Y=None, do_backward=False):
        if X is not None:
            K = self.K
            n_batch = X.shape[1]
            return generate_random_one_hot(K,n_batch)

    def set_params(self, params: List[np.ndarray]):
        self._params = params