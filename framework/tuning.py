import random
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class Tuner(ABC):
    @abstractmethod
    def optimize(self, hypermodel, system_evaluator, increase=True,
                 return_top = 1):
        """
        hypermodel: A function that takes a dict as input and builds
                    a model

        system_evaluator: A function that takes a model as input
                    and returns a metric score to optimize

        increase: The direction of the optimization

        return_top: If specified will return the top solutions of
                    this number
        """
        pass

class RandomTuner(Tuner):
    def __init__(self, base_config,
                 tunable_config : Dict[str, "FloatRange"],
                 n_tries):
        """
        base_config: Key-value pairs for input to the model
        constructor these parameters will be static for every model
        creation

        tunable_config: Key-value of range pairs that will
        be tuned hypermodel must be a function that can take a
        base_config and create a model

        """
        self.base_config = base_config
        self.tunable_config = tunable_config

        self.n_tries = n_tries

    def optimize(self, hypermodel, system_evaluator, increase=True,
                 return_top=1):
        res = []
        for i in range(self.n_tries):
            sampled_params = {k : v.sample() for k,v in
                              self.tunable_config.items()}
            # TODO How do i tune data augmentation
            config = {**self.base_config, **self.tunable_config}
            model = hypermodel(**config)
            metric = system_evaluator(model)
            res.append((config, metric))

        return sorted(res, key=lambda x : x[1], reverse=not increase,
                      )[:return_top]


class FloatRange:
    def __init__(self, start, end, log_sample=False):
        self.start =start
        self.end = end
        self.log_sample = log_sample
        self.transform = lambda x : x
        # If its log we uniformly sample the exponent
        if self.log_sample:
            self.start = np.log(start)
            self.end = np.log10(end)
            self.transform = lambda x : 10 ** x

    def sample(self):
        # Uniform sample from range
        res = self.start + random.random() * (self.end - self.start)
        return self.transform(res)