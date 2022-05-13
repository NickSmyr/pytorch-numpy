import unittest

from framework.gradcheck import gradcheck
from framework.model import twolayer, ReLuDropoutBatchNormMLP, VanillaRNN
from framework.non_pure import deterministic_seed
from test.convenience import generate_random_data_set, \
    generate_random_text_data_set


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        deterministic_seed(2)

    def test_grads_2layer_model(self):
        n_batch = 5
        input_dim = 3072
        K = 10
        hidden_dim = 6
        startIdx = 3
        endIdx = 15
        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = twolayer(K, input_dim, hidden_dim, 1, 0.6, 0.6, 0.6)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_grads_2layer_mlp(self):
        n_batch = 5
        input_dim = 3072
        K = 10
        hidden_dim = 6
        startIdx = 3
        endIdx = 15
        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = ReLuDropoutBatchNormMLP(input_dim, K, [hidden_dim], 1, 0.6,
                                        0.6, 0.6)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_grads_2layer_mlp_with_batch_norm(self):
        n_batch = 5
        n_channels = 2
        n_width = 3
        n_height = 4
        input_dim = n_channels * n_width * n_height
        K = 10
        hidden_dim = 6

        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = ReLuDropoutBatchNormMLP(input_dim, K, [hidden_dim], 1, 0,
                                        0.6, 0.6, num_channels=n_channels,
                                        image_width=n_width,
                                        image_height=n_height,
                                        use_batchnorm=True)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    # TODO For determnistic seed 6, for the first input batch item,
    #   all activations In the first layer become zero leading to some
    #   gradient errors

    def test_grads_3layer_mlp(self):
        n_batch = 5
        input_dim = 3072
        K = 10
        hidden_dims = [6, 5]

        lamda = 0.001

        pdrop = 0.1
        pflip = 0.2
        ptrans = 0.2

        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = ReLuDropoutBatchNormMLP(input_dim, K, hidden_dims, lamda,
                                        pdrop, pflip,
                                        ptrans)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_grads_3layer_mlp_with_batchnorm(self):
        n_batch = 5
        input_dim = 3072
        K = 10
        hidden_dims = [6, 5]

        lamda = 0.001

        pdrop = 0.1
        pflip = 0.2
        ptrans = 0.2

        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = ReLuDropoutBatchNormMLP(input_dim, K, hidden_dims, lamda,
                                        pdrop, pflip, ptrans,
                                        use_batchnorm=True)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_grads_4layer_mlp(self):
        n_batch = 5
        input_dim = 3072
        K = 10
        hidden_dims = [6, 5, 4]

        X, Y = generate_random_data_set(K, input_dim, n_batch)

        model = ReLuDropoutBatchNormMLP(input_dim, K, hidden_dims, 0.001,
                                        0.1, 0.2, 0.4)
        res = gradcheck(X, Y, model, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_vanilla_rnn_generates_random_seqs(self):
        element_set = {'a', 'b', 'c'}
        input_dim = 3
        output_dim = 3
        hidden_dim = 50
        num_steps = 50
        rnn = VanillaRNN(input_dim, hidden_dim, output_dim, num_steps,
                         element_set)

        res = rnn.generate_sequence("a", 10)
        self.assertTrue(type(res) == str)

    def test_vanilla_rnn_grads(self):
        element_set = set(list(range(72)))
        input_dim = 72
        output_dim = 72
        hidden_dim = 50
        num_steps = 16
        batch_size = 4

        rnn = VanillaRNN(input_dim, hidden_dim, output_dim, num_steps,
                         element_set)
        X, Y = generate_random_text_data_set(element_set, num_steps,
                                             batch_size)

        res = gradcheck(X, Y, rnn, 1e-6, 1e-5)
        self.assertTrue(res)

    def test_grads_vanilla_rnn(self):
        pass


if __name__ == '__main__':
    unittest.main()
