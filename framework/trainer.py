import copy
from typing import Tuple, Union, List

import numpy as np
import pandas as pd

from framework.augmentation import BatchAugmenter
from framework.evaluator import Evaluator
from framework.model import Model, VanillaRNN, print_param_grad_summary

from tqdm import tqdm


def batches(seq, size):
    return [seq[:, pos:pos + size] for pos in range(0, seq.shape[1], size)]


class Trainer:
    def __init__(self,
                 model: Model,
                 trainX,
                 trainY,
                 valX=None,
                 valY=None,
                 lr=0.001,
                 lr_scheduler: "LearningRateSchedule" = None,
                 n_epochs: int = None,
                 n_cycles: int = None,
                 batch_size: int = None,
                 reshuffle_on_epoch_start: bool = False,
                 optimizer="sgd"):

        self.model = model

        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY

        self.lr = lr
        self.lr_scheduler = lr_scheduler

        self.n_epochs = n_epochs
        self.n_cycles = n_cycles

        self.batch_size = batch_size
        self.epoch_end_evaluators = []

        self.reshuffle_on_epoch_start = reshuffle_on_epoch_start
        self.optimizer = optimizer
        if self.optimizer == "adam" or self.optimizer == "adagrad":
            self.adam_ms = [np.zeros_like(x) for x in self.model.params]
            self.adam_vs = [np.zeros_like(x) for x in self.model.params]

    def attach_evaluators_on_epoch_end(self,
                                       evaluators: Union[
                                           Tuple[str, Evaluator], List[Tuple[
                                               str, Evaluator]]]):
        """
        Input: One or more pairs of evaluator_name and Evaluator objects
        """
        if not isinstance(evaluators, list):
            evaluators = [evaluators]

        self.epoch_end_evaluators = evaluators

    def train_batch(self, x_batch, y_batch, step_i):

        # Calculate step learning rate
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.lr_at_step(step_i)
        else:
            lr = self.lr

        self.model.zero_grads()
        loss = self.model.forward(x_batch, y_batch, do_backward=True)

        model_grads = self.model.grads
        old_params = self.model.params

        if self.optimizer == "sgd":
            new_params = []
            for param, grad in zip(old_params, model_grads):
                new_param = param - lr * grad
                new_params.append(new_param)

            self.model.set_params(new_params)

        elif self.optimizer == "adam":
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            new_params = []
            for i, (param, grad) in enumerate(zip(old_params, model_grads)):
                # Clip grad
                grad = np.minimum(np.ones_like(grad) *  5, grad)
                grad = np.maximum(np.ones_like(grad) * -5, grad)

                self.adam_ms[i] = beta1 * self.adam_ms[i] + (1-beta1) * grad
                self.adam_vs[i] = beta2 * self.adam_vs[i] + (1 - beta2) * (
                    grad * grad)

                # rescale
                m_rescaled = self.adam_ms[i] / (1 - (beta1 ** (step_i + 1)))
                v_rescaled = self.adam_vs[i] / (1 - (beta2 ** (step_i + 1)))

                #update
                new_param = param  - (lr / (np.sqrt(v_rescaled) + eps)) * \
                            m_rescaled

                new_params.append(new_param)

            self.model.set_params(new_params)
        elif self.optimizer == "adagrad":
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            new_params = []
            for i, (param, grad) in enumerate(zip(old_params, model_grads)):
                # Clip grad
                grad = np.minimum(np.ones_like(grad) *  5, grad)
                grad = np.maximum(np.ones_like(grad) * -5, grad)

                self.adam_ms[i] = self.adam_ms[i] + (grad**2)

                #update
                new_param = param  - (lr / (np.sqrt(self.adam_ms[i]+ eps)))\
                            * \
                            grad

                new_params.append(new_param)
            self.model.set_params(new_params)
        else:
            raise Exception("Optimizer must be one sgd or adam")

        return loss

    def on_epoch_start(self):
        """
        Called when the epoch starts
        """
        # Randomize train set at the start of each epoch
        if self.reshuffle_on_epoch_start:
            random_shuffle_idxs = np.random.permutation(range(
                self.trainX.shape[1]))
            self.trainX = self.trainX[:, random_shuffle_idxs]
            self.trainY = self.trainY[:, random_shuffle_idxs]

    def train(self):
        """
        :return: A dict of evaluation results Dict[str, List] for every on
        epoch
        end evaluator evaluated at every epoch
        """
        self.on_training_start()
        # Set the model in train mode
        self.model.set_training()

        if self.n_epochs is None:
            if isinstance(self.lr_scheduler,
                          CyclicalLearningRateSchedule):
                num_steps = self.lr_scheduler.get_cycle_len() * \
                            self.n_cycles
            else:
                raise Exception("A non cyclical learning schedule"
                                " requires the n_epochs argument")
        else:
            num_steps = self.n_epochs * len(self.train_batches)

        pbar = tqdm(total=num_steps)
        self.step_i = 0
        done = False
        pass
        while not done:
            self.on_epoch_start()
            for xb, yb in self.train_batches:
                self.last_batch_loss = self.train_batch(xb, yb, self.step_i)
                self.on_train_batch_end()
                self.step_i += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {self.last_batch_loss}")

                if self.step_i == num_steps:
                    done = True
                    break

            self.on_epoch_end()

        pbar.close()
        return self.return_from_train()

    def return_from_train(self):
        return self.eval_return

    def on_train_batch_end(self):
        pass

    def on_training_start(self):
        # Initialize return result
        self.eval_return = {}
        for target in ["train", "val"]:
            self.eval_return[target] = {}
            for eval_name, evaluator in self.epoch_end_evaluators:
                self.eval_return[target][eval_name] = []

        self.train_batches = list(zip(batches(self.trainX,
                                              self.batch_size),
                                      batches(self.trainY,
                                              self.batch_size)))

    def on_epoch_end(self):
        # Set the model in test mode
        self.model.set_test()
        # Evaluate at the end of the epoch
        # Restrain evaluators to be on certain targets (val, test)
        Y_pred_train = self.model.forward(self.trainX)
        Y_pred_val = self.model.forward(self.valX)
        for eval_name, evaluator in self.epoch_end_evaluators:
            for target in ['train', 'val']:
                dataset = (self.valX, self.valY) if target \
                                                    == "val" else \
                    (self.trainX, self.trainY)
                probas = Y_pred_train if target == "train" else \
                    Y_pred_val
                self.eval_return[target][eval_name].append(
                    evaluator.evaluate(self.model, *dataset, probas
                                       ))


class RNNTrainer(Trainer):
    def return_from_train(self):
        model : VanillaRNN = self.model
        pd.DataFrame(self.smooth_loss_list).to_csv("smooth_loss.csv")
        print("Best model loss ", self.best_loss)
        print("Printing best model generation of 1000 chars")
        model.reset_hidden_state()
        seq = model.generate_sequence('.', 200)
        print(seq)
        return None

    def on_training_start(self):
        self.train_batches = [(x,y) for x,y in zip(self.trainX, self.trainY)]

    def on_epoch_end(self):
        pass

    def on_train_batch_end(self):
        model : VanillaRNN = self.model

        if "smooth_loss" not in self.__dict__:
            self.smooth_loss = self.last_batch_loss
            self.smooth_loss_list = [self.last_batch_loss]

            self.best_loss = self.last_batch_loss
            self.best_model = copy.deepcopy(self.model)

        else:
            self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * self.last_batch_loss
            self.smooth_loss_list.append(self.smooth_loss)

        if self.step_i % 1000 == 0:
            print("Iter = {}, smooth_loss = {}".format(self.step_i,
                                                       self.smooth_loss))
            model_hidden = model.forward_last_hidden
            model.reset_hidden_state()
            seq = model.generate_sequence('.', 200)
            print("Train batch end summ")
            print_param_grad_summary(model)

            print("Printing a generated sequence")
            print(seq)
            model.set_hidden_for_next_formard(model_hidden)

        else:
            model.set_hidden_for_next_formard(model.forward_last_hidden)


class LearningRateSchedule:
    def lr_at_step(self, step_i):
        """
        Get lr at step
        """
        raise NotImplementedError


class CyclicalLearningRateSchedule(LearningRateSchedule):
    def get_cycle_len(self):
        """
        Get length of cycle
        """
        raise NotImplementedError


class TriangularLRSchedule(CyclicalLearningRateSchedule):
    def __init__(self, eta_min, eta_max, n_s):
        """
        n_s is half the cycle len
        """
        self.n_s = n_s
        self.eta_min = eta_min
        self.eta_max = eta_max

    def get_cycle_len(self):
        return 2 * self.n_s

    def lr_at_step(self, step_i):
        x = step_i // self.n_s
        # print("T is ", t, "L is ", l)
        if x % 2 == 0:
            l = x / 2
            return self.eta_min + (
                        (step_i - 2 * l * self.n_s) / (self.n_s)) * (
                               self.eta_max -
                               self.eta_min)
        else:
            l = (x - 1) / 2
            return self.eta_max - (
                        (step_i - (2 * l + 1) * self.n_s) / (self.n_s)) * (
                           self.eta_max -
                           self.eta_min)
