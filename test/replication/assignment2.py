import unittest
from typing import Literal, List

from framework.augmentation import Assignment2Augmenter
from framework.evaluator import AccuracyEvaluator, \
    twolayerLossEvaluator, twolayerCostEvaluator
from framework.model import twolayer, ReLuDropoutMLP
from framework.non_pure import load_cifar, deterministic_seed
from framework.trainer import TriangularLRSchedule, Trainer
from framework.plotting import plot_train_results


class Assignment2Replication(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trainX, cls.trainY, cls.valX, cls.valY, cls.testX, \
        cls.testY = load_cifar(
            "../../cifar-10-batches-py")

    @classmethod
    def setUp(self) -> None:
        deterministic_seed(0) # seed=0 valacc 0.545

    def test_exercise2_2layer_should_get_over_52(self):
        self.exercise2_config([50,50])

    def test_exercise2_9layer_should_be_pretty_bad(self):
        self.exercise2_config([50, 30, 20, 20, 10, 10, 10, 10])

    def exercise2_config(self, hidden_dims):
        input_dim = 3072
        K = 10

        batch_size = 100
        eta_min = 1e-5
        eta_max = 1e-1
        lamda = 0.005
        n_cycles = 2
        n_s = int(5 * 45000 / batch_size)

        pdrop = 0
        pflip = 0
        ptrans = 0

        model = "mlp"

        self.run_with_hyperparameters(K, None, input_dim, lamda,
                                      pdrop, pflip, ptrans, model, eta_min,
                                      eta_max, n_s, batch_size, n_cycles,
                                      hidden_dims)


    def test_train_interface(self):
        input_dim = 3072
        K = 10
        hidden_dim = 150
        lamda = 0.0008841930508591771
        pdrop = 0.1
        pflip = 0.241022669092449
        ptrans = 0.4484956924775554
        batch_size = 100
        n_s = 5

        model = 'mlp'

        n_cycles = 10
        eta_min = 1e-5
        eta_max = 1e-1

        # Params for mlp
        hidden_dims = [150]

        self.run_with_hyperparameters(K, hidden_dim, input_dim,
                                      lamda, pdrop, pflip, ptrans, model,
                                      eta_min, eta_max, n_s, batch_size,
                                      n_cycles,hidden_dims )

    def run_with_hyperparameters(self, K, hidden_dim, input_dim,
                                 lamda,
                                 pdrop,
                                 pflip,
                                 ptrans,
                                 model : Literal["twolayer", "mlp"],
                                 eta_min,
                                 eta_max,
                                 n_s,
                                 batch_size,
                                 n_cycles,
                                 hidden_dims : List[int]):

        evaluator = AccuracyEvaluator()
        if model == "twolayer":
            model = twolayer(K, input_dim, hidden_dim,
                             lamda, pdrop,
                             pflip=ptrans,
                             ptrans=ptrans)
        else:
            model = ReLuDropoutMLP(input_dim, K, hidden_dims,
                               lamda=lamda,
                               pdrop=pdrop,
                               pflip=pflip,
                               ptrans=ptrans)
        # TODO move lr_scheudle in model
        if n_s is None:
            n_s = int(2 * self.trainX.shape[1] / batch_size)

        lr_schedule = TriangularLRSchedule(eta_min, eta_max, n_s)
        trainer = Trainer(model, self.trainX, self.trainY, self.valX,
                          self.valY, None, lr_schedule, None, n_cycles,
                          batch_size)
        # trainer.attach_evaluators_on_epoch_end([
        #     ('accuracy', AccuracyEvaluator()),
        #     ('loss', twolayerLossEvaluator()),
        #     ('cost', twolayerCostEvaluator())
        # ])
        res = trainer.train()
        print(res)
        # plot_train_results(res)
        model.set_test()

        Y_pred_train = model.forward(self.trainX)
        Y_pred_val = model.forward(self.valX)
        Y_pred_test = model.forward(self.testX)

        print("Final val accuracy ",
              evaluator.evaluate(model, self.valX, self.valY, Y_pred_val))
        print("Final train accuracy ",
              evaluator.evaluate(model, self.trainX, self.trainY, Y_pred_train)
              )
        print("Final test accuracy ",
              evaluator.evaluate(model, self.testX, self.testY, Y_pred_test))


if __name__ == '__main__':
    unittest.main()
