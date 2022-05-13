import unittest

from framework.augmentation import FlipImage
from framework.evaluator import AccuracyEvaluator, \
    twolayerLossEvaluator
from framework.gradcheck import gradcheck
from framework.model import twolayer, ReLuDropoutBatchNormMLP, VanillaRNN
from framework.non_pure import deterministic_seed, load_cifar, \
    batch_chars_dataset
from framework.pure import generate_dropout_mask
import numpy as np

from framework.trainer import TriangularLRSchedule, Trainer, RNNTrainer
from test.convenience import generate_random_data_set
from test.stubs import RandomModel


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.trainX, cls.trainY, cls.valX, cls.valY, cls.testX, \
        cls.testY = load_cifar(
            "../../cifar-10-batches-py")

    def setUp(self) -> None:
        deterministic_seed(8)

    # TODO this is necessary for refactoring
    def test_training_returns_specified_outputs(self):
        n_batch = 4
        K = 5

        num_channels = 3
        image_width = 2
        image_height = 2

        input_dim = num_channels * image_height * image_width
        hidden_dim = 7

        trainX, trainY = generate_random_data_set(K, input_dim,
                                                 n_batch)
        valX, valY = generate_random_data_set(K, input_dim, n_batch)

        model = twolayer(K, input_dim, hidden_dim, 0.001, 0.1, 0.3,
                         0.5, num_channels=num_channels,
                         image_height=image_height,
                         image_width=image_width)

        trainer = Trainer(model, trainX, trainY, valX, valY, 0.001,
                          n_epochs=2, batch_size=2,
                          reshuffle_on_epoch_start=True)
        evaluators = [
            ('accuracy', AccuracyEvaluator()),
            ('loss' , twolayerLossEvaluator())
        ]
        trainer.attach_evaluators_on_epoch_end(evaluators)
        results = trainer.train()
        keys = [
            "accuracy",
            "loss"
        ]
        for target in ['val', 'train']:
            for k in keys:
                self.assertTrue(k in results[target])
                self.assertTrue(isinstance(results[target][k], list))
                self.assertTrue(len(results[target][k]) == 2)




    def test_dropout(self):
        drop = generate_dropout_mask(0.7, (5,15))
        self.assertEqual(drop.shape, (5,15))

    def test_deterministic_dropout(self):
        n_batch = 3

        num_channels = 3
        image_width = 2
        image_height = 2

        input_dim = num_channels * image_width * image_height

        K = 10
        hidden_dim = 30

        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = twolayer(K, input_dim, hidden_dim, 1, 0.5, 0.5, 0.5,
                         num_channels, image_width, image_height
                         )

        deterministic_seed(0)
        loss1 = model.forward(X, Y, do_backward=True)
        deterministic_seed(0)
        loss2 = model.forward(X,Y, do_backward=True)
        self.assertEqual(loss1, loss2)

    def test_randomness_of_dropout(self):
        """Super flaky"""
        dropout_shape = (50,100)
        deterministic_seed(0)
        mask1 = generate_dropout_mask(0.5,  dropout_shape)
        deterministic_seed(1)
        mask2 = generate_dropout_mask(0.5, dropout_shape)
        self.assertFalse((mask1 == mask2).all())\

    # TODO (NOT URGENT): THat network overfits

    def test_triangular_lr(self):
        eta_min = 1e-7
        eta_max = 1
        n_s = 500
        sch = TriangularLRSchedule(eta_min, eta_max, n_s)
        self.assertTrue(sch.get_cycle_len() == 2*n_s)
        self.assertTrue(sch.lr_at_step(0), eta_min)
        self.assertTrue(sch.lr_at_step(n_s), eta_max)
        self.assertTrue(sch.lr_at_step(2*n_s), eta_min)

    def test_accuracy(self):
        eval = AccuracyEvaluator()
        X,Y = generate_random_data_set(10, 5 , 3)
        model = RandomModel(10)
        res = eval.evaluate(model, X, Y, model.forward(X))
        print("Res is ", res)
        self.assertTrue(res <= 1 and res >= 0)

    def test_flip(self):
        img_shape = (3,32,32,1)
        img = self.trainX[:, 2:3].reshape(img_shape)
        trans = FlipImage()
        flipped = trans.apply(img)
        double_flipped = trans.apply(flipped)
        self.assertTrue((double_flipped == img).all())

    def test_can_train_vanilla_rnn(self):
        char_list = 'asv.awe'*10000
        char_set = set(char_list)
        X_batches, Y_batches = batch_chars_dataset(char_list, 13, 14,
                                                   stride=5,
                                                   drop_last_batch=True)
        model = VanillaRNN(len(char_set), 50, len(char_set), 30,
                           element_set=char_set)
        trainer = RNNTrainer(model, X_batches, Y_batches, lr=0.001,
                             n_epochs=1, batch_size=1,
                             reshuffle_on_epoch_start=False, optimizer="adam")
        trainer.train()


if __name__ == '__main__':
    unittest.main()
