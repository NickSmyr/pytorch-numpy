import math
import sys

from .layer import Layer
from .model import Model
import numpy as np

from .non_pure import deterministic_seed


def gradcheck(X, Y, model : Model, h : float, tol: float):
    model.zero_grads()
    # Set the same seed before forward
    deterministic_seed(0)
    model.forward(X,Y, do_backward=True)
    true_grads = model.grads
    params_copy = [np.copy(x) for x in model.params]

    # For param in parameters
    #   For idx in nditer in parameters
    #       b4loss = x
    #       idxtry +=h
    #       a8loss = x'
    #       grad_param_idx = ...
    #       if diff > tol: raise Error
    max_diff = -math.inf
    params_diffs = []
    for i, param in enumerate(model.params):
        it = np.nditer(param, flags=['multi_index'])
        param_diffs = []
        for _ in it:
            idx = it.multi_index
            param_idx_before = param[idx]
            # Left
            params_copy[i][idx] = param_idx_before -h
            model.set_params(params_copy)
            # Set the same seed before forward
            deterministic_seed(0)
            # We want to also do the backward pass because of
            # data augmentation
            c1 = model.forward(X,Y, do_backward=True)

            # Right
            params_copy[i][idx] = param_idx_before + h
            model.set_params(params_copy)
            # Set the same seed before forward
            deterministic_seed(0)
            c2 = model.forward(X, Y, do_backward=True)
            # Reset val
            params_copy[i][idx] = param_idx_before

            grad_calc = (c2 - c1) / (2*h)
            diff = abs(grad_calc - true_grads[i][idx])
            relerror = abs((grad_calc - true_grads[i][idx])/grad_calc)
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                print(f"Grad check failed at index {i} with diff: ",
                      diff, "and element ", idx, "true grad:" , grad_calc,
                      file=sys.stderr)
                pass
            param_diffs.append(diff)
        params_diffs.append(param_diffs)

    parameter_names = model.param_names()
    print("Gradcheck passed with max diff ", max_diff)
    print("Max absolute diffs per param ",
          {name : max(param) for param, name in zip(params_diffs,
                                                parameter_names)})

    return max_diff < tol


def gradcheck_layer(X, layer : Layer, h, tol):
    """
    Layer is check with the MSE loss (y**2/2)
    """

    params_copy = [np.copy(x) for x in layer.params]
    def mse_loss(x):
        #return np.sum(x**2)/2
        return np.sum(x) / x.shape[1]

    layer.zero_grads()
    # Set the same seed before forward
    deterministic_seed(0)
    layer_out = layer.forward(X)
    gradients_out = layer.backward(np.ones_like(layer_out))
    true_grads = layer.grads

    max_diff = -math.inf
    params_diffs = []
    for i, param in enumerate(layer.params):
        it = np.nditer(param, flags=['multi_index'])
        param_diffs = []
        for _ in it:
            idx = it.multi_index
            param_idx_before = param[idx]
            # Left
            params_copy[i][idx] = param_idx_before - h
            layer.set_params(params_copy)
            # Set the same seed before forward
            deterministic_seed(0)
            c1 = mse_loss(layer.forward(X))

            # Right
            params_copy[i][idx] = param_idx_before + h
            layer.set_params(params_copy)
            # Set the same seed before forward
            deterministic_seed(0)
            c2 = mse_loss(layer.forward(X))
            # Reset val
            params_copy[i][idx] = param_idx_before

            grad_calc = (c2 - c1) / (2 * h)
            diff = abs(grad_calc - true_grads[i][idx])
            if diff > max_diff:
                max_diff = diff
            if diff > tol:
                # print(f"Grad check failed at index {i} with diff: ",
                # diff, file=sys.stderr)
                pass
            param_diffs.append(diff)
        params_diffs.append(param_diffs)

    ## TODO check that the output_gradients are correct

    print("Gradcheck passed with max diff ", max_diff)
    print("Max absolute diffs per param ",
          {pname : max(p) for p, pname in zip(params_diffs,
                                              layer.param_names)})

    return max_diff < tol
