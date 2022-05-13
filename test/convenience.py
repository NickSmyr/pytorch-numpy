import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt

def hash_np_array(x : np.ndarray):
    return hash(x.data.tobytes())

def zeros_like_list(list : List[np.ndarray]):
    """Create a list of zero_like tensors from input list"""
    return [np.zeros_like(x) for x in list]

def np_copy_list(list : List[np.array]):
    return [np.copy(x) for x in list]

def generate_random_shape(shape):
    return np.random.randn(*shape)

def generate_random_one_hot(K , n_batch):
    random_idxs = random.choices(list(range(K)), k=n_batch)
    return np.eye(K,K)[random_idxs].T

def plot_cifar_image(x):
    plt.imshow(x.transpose([1,2,0]))
    plt.show()

def plot_cifar_image_batch(x):
    plt.imshow(x[...,0].transpose([1,2,0]))
    plt.show()

def generate_random_data_set(K, dim, n_batch):
    return generate_random_shape((dim, n_batch)), generate_random_one_hot(K,n_batch)


def generate_random_text_data_set(element_set, n_seq, n_batch):
    return np.random.choice(list(element_set), (n_seq, n_batch)), \
           np.random.choice(list(element_set), (n_seq, n_batch))


