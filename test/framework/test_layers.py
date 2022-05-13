import unittest

import numpy as np

from framework.gradcheck import gradcheck, gradcheck_layer
from framework.layer import Linear, ReLu, Sequential, Dropout, BatchNorm, \
    Residual, OneHotEmbeddingLayer, Tanh
from framework.non_pure import deterministic_seed
from test.convenience import generate_random_data_set, hash_np_array


class MyTestCase(unittest.TestCase):
    ## TODO after this create framework for testing layers
    def setUp(self) -> None:
        deterministic_seed(0)
        # Generate common data
        self.input_dim = 10
        self.output_dim = 20
        self.K = 5
        self.n_batch = 4
        self.lamda = 0.5
        self.X, self.Y = generate_random_data_set(self.K,
                                                  self.input_dim,
                                                 self.n_batch)

    def test_accumulation_of_grads_seq(self):
        hidden_dim = 6
        layer = Sequential(
            [
                Linear(self.input_dim, hidden_dim, 0),
                ReLu(),
                Dropout(pdrop=0.6),
                Linear(hidden_dim, self.K, 0)
            ]
        )
        deterministic_seed(0)
        _ = layer.forward(self.X)
        deterministic_seed(0)
        _ = layer.forward(self.X)
        layer.backward(np.ones((self.K,  self.X.shape[1])))
        prev_grads = [grad for grad in layer.grads]
        # Do another backward
        layer.backward(np.ones((self.K, self.X.shape[1])))
        new_grads = [grad for grad in layer.grads]
        for prev, new in zip(prev_grads, new_grads):
            self.assertTrue((2 * prev == new).all())


    def test_embedding_correct_forward(self):
        elements = {'a', 'b', 'c'}
        layer = OneHotEmbeddingLayer(set_of_elements=elements)
        layer_input = np.array([
            ['a', 'c'],
            ['c', 'a'],
            ['b', 'b']
        ])
        res = layer.forward(layer_input)
        # embedding size x seq_length, batch_size
        self.assertTrue(res.shape == (3, 3, 2))

    def test_embedding_decoding(self):
        elements = {'a', 'b', 'c'}
        layer = OneHotEmbeddingLayer(set_of_elements=elements)
        layer_input = np.array([
            [0, 2],
            [2, 0],
            [1, 1]
        ])
        output_tens = layer.decode(layer_input)
        self.assertTrue(output_tens.shape == (3, 2))

    def test_sequential_gradcheck(self):
        hidden_dim = 6
        layer = Sequential(
            [
                Linear(self.input_dim, hidden_dim, 0),
                ReLu(),
                Dropout(pdrop=0.6),
                Linear(hidden_dim, self.K, 0)
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_residual_gradcheck(self):
        hidden_dims = [30, 30]
        layer = Sequential(
            [
                Linear(self.input_dim, hidden_dims[0], 0),
                ReLu(),
                Dropout(pdrop=0.6),
                Residual(Sequential(
                    [
                        Linear(hidden_dims[0], hidden_dims[1], 0),
                        ReLu(),
                        Dropout(pdrop=0.6),
                    ]
                )),
                Linear(hidden_dims[-1], self.K, 0)
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    # This fails because an items activations become completely zero
    def test_sequential_3_layer_gradcheck(self):
        hidden_dim = 10
        pdrop = 0.6
        layer = Sequential(
            [
                Linear(self.input_dim, hidden_dim, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim, hidden_dim-2, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim - 2, hidden_dim - 2, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim - 2, hidden_dim - 2, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim - 2, hidden_dim - 2, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim - 2, hidden_dim - 2, 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dim-2, self.K, 0)
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_sequential_3_layer_gradcheck_2(self):
        hidden_dims = [150, 150]
        pdrop = 0.6
        layer = Sequential(
            [
                Linear(self.input_dim, hidden_dims[0], 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dims[0], hidden_dims[1], 0),
                ReLu(),
                Dropout(pdrop=pdrop),
                Linear(hidden_dims[1], self.output_dim, 0),
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_dropout_gradcheck(self):
        layer = Sequential(
            [
                Linear(self.input_dim, self.K, 0),
                Dropout(pdrop=0.6),
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_relu_gradcheck(self):
        layer = Sequential(
            [
                Linear(self.input_dim, self.K, 0),
                ReLu()
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_tanh_gradcheck(self):
        layer = Sequential(
            [
                Linear(self.input_dim, self.K, 0),
                Tanh()
            ]
        )
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_batchnorm_gradcheck(self):
        layer = Sequential([
            Linear(self.input_dim, self.K, 0),
            BatchNorm(self.K, 0.9)
        ])
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)


    def test_batchnorm_gradcheck_3_layer(self):
        layer = Sequential([
            Linear(self.input_dim, self.K, 0),
            BatchNorm(self.K, 0.9),
            ReLu(),
            Linear(self.K, self.K, 0),
            BatchNorm(self.K, 0.9),
            ReLu(),
            Linear(self.K, self.K, 0),
            BatchNorm(self.K, 0.9),
        ])
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    def test_linear_gradcheck(self):
        layer = Linear(self.input_dim, self.K, 0)
        res = gradcheck_layer(self.X, layer, 1e-5, 1e-6)
        self.assertTrue(res)

    # TODO Fix this
    # def test_linear(self):
    #     input_dim = 10
    #     K = 5
    #     n_batch = 4
    #     X,Y = generate_random_data_set(K, input_dim, n_batch)
    #
    #     layer = Linear(input_dim, K, 2)
    #     layer_out = layer.forward(X) #K,n_batch
    #     self.assertTrue(layer_out.shape == (K, n_batch))
    #
    #     # Test MSE loss
    #     # true_loss = (layer_out)**2 / 2
    #     true_grads= 2* layer.params[0], 2*np.ones_like(
    #         layer.params[1])
    #     input_grads = np.ones_like(layer_out)
    #     # Do backward
    #     layer.backward(input_grads)
    #     calc_grads = layer.grads
    #     # Assertion for W
    #     diffs = np.abs(true_grads[0] - calc_grads[0])
    #     self.assertTrue(np.allclose(true_grads[0], calc_grads[0],
    #                                 atol=1e-6))
    #     # Assertion for b
    #     self.assertTrue(np.allclose(true_grads[1], calc_grads[1],
    #                                 atol=1e-6))


if __name__ == '__main__':
    unittest.main()
