import numpy as np
def softmax(x, ax=0, temperature=1):
    x = x / temperature
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return (p / np.sum(p, axis=ax, keepdims=True))

def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

def relu(x):
    return (abs(x) + x) / 2

def tanh(x):
    return np.tanh(x)

def generate_dropout_mask(pdrop, shape):
    dropout = np.random.binomial(1, 1 - pdrop, shape)
    return dropout