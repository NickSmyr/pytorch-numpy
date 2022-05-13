import numpy as np
import random
from six.moves import cPickle


# TODO not working
from tqdm import tqdm


def deterministic_seed(seed=None):
    """
    Wrapper when we are using multiple number generators
    """
    np.random.seed(seed)
    random.seed(seed)

def pload(fname):
    with open(fname, "rb") as f:
        return cPickle.load(f,encoding='latin1')


def load_cifar(cifar_path):
    train_data_fnames = [f"{cifar_path}/data_batch_{x}" for x in range(1, 6)]
    test_data_fname = f"{cifar_path}/test_batch"

    train_data_arrays = [pload(x) for x in train_data_fnames]
    test_data_array = pload(test_data_fname)

    trainX = np.concatenate([x['data'] for x in train_data_arrays])
    trainY = np.concatenate([x['labels'] for x in train_data_arrays])

    ## Data logic for assignment 1 (We don't use the entire dataset)
    # trainX = np.concatenate([x['data'] for x in [train_data_arrays[0]]])
    # trainY = np.concatenate([x['labels'] for x in [train_data_arrays[0]]])

    valX = np.concatenate([x['data'] for x in [train_data_arrays[1]]])
    valY = np.concatenate([x['labels'] for x in [train_data_arrays[1]]])

    eye = np.eye(10, 10)
    valY = eye[valY]
    train_mean = np.mean(trainX, axis=0)
    train_std = np.mean(trainX, axis=0)
    valX = (valX - train_mean) / train_std

    testX = test_data_array['data']
    testY = np.array(test_data_array['labels'])

    eye = np.eye(10, 10)
    # Convert Y data to one hot
    trainY = eye[trainY]
    testY = eye[testY]

    # Normalize
    train_mean = np.mean(trainX, axis=0)
    train_std = np.mean(trainX, axis=0)

    trainX = (trainX - train_mean) / train_std
    testX = (testX - train_mean) / train_std

    # Split to trainval
    indxs = set(range(trainX.shape[0]))
    val_indxs = set(random.sample(list(indxs), k=1000))
    train_indxs = list(indxs - val_indxs)
    val_indxs = list(val_indxs)
    valX = trainX[val_indxs, :]
    valY = trainY[val_indxs, :]
    trainX = trainX[train_indxs, :]
    trainY = trainY[train_indxs, :]

    # Transpose
    trainX = trainX.T
    valX = valX.T
    testX = testX.T

    trainY = trainY.T
    valY = valY.T
    testY = testY.T

    return trainX, trainY, valX, valY, testX, testY

def load_goblet_of_fire(text_file_path):
    with open(text_file_path) as f:
        lines = f.readlines()

    translation_table = dict.fromkeys(map(ord, '\t\n'), None)
    string_patterns_to_skip = ["CHAPTER",
                               "HARRY POTTER AND THE GOBLET OF FIRE"]

    new_lines = []
    for line in lines:
        if all([pat not in line for pat in string_patterns_to_skip]):
            new_lines.append(line.translate(translation_table))
    lines = new_lines

    lines = " ".join(lines)

    # Remove extra whitespace between words

    char_list = " ".join(lines.split())

    chars =  sorted(set(lines))
    return char_list,  chars

def batch_chars_dataset(char_list, batch_size, seq_length,
                        stride, drop_last_batch):
    input_char_seqs = []
    output_char_seqs = []
    for pos in range(0, len(char_list), stride):
        # Dont add less that seq length sequence
        if pos + seq_length + 1 > len(char_list):
            break
        input_list = list(char_list[pos : pos + seq_length])
        output_list = list(char_list[pos + 1 : pos + seq_length + 1])

        input_char_seqs.append(input_list)
        output_char_seqs.append(output_list)

    print("INput char seqs")
    print("".join(input_char_seqs[0]))

    input_batches = []
    output_batches = []
    print("Batching data")
    # Split sequences to bs number of continuous chunks
    for i in range(0, len(input_char_seqs) // batch_size):
        # Want input_batches[0][:,0] continues with input_batches[0][:,1]
        curr_xb = input_char_seqs[i:i +batch_size * batch_size:batch_size]
        # x bs
        curr_yb = output_char_seqs[i:i +batch_size * batch_size:batch_size]

        input_batches.append(np.array(curr_xb).T)
        output_batches.append(np.array(curr_yb).T)
    # input_batches = [ np.array(input_char_seqs[pos : pos + batch_size]).T
    #             for pos in range(0, len(input_char_seqs), batch_size)]
    # output_batches = [np.array(output_char_seqs[pos: pos + batch_size]).T
    #                  for pos in range(0, len(output_char_seqs), batch_size)]
    if drop_last_batch:
        input_batches = input_batches[:-1]
        output_batches = output_batches[:-1]

    return input_batches, output_batches



if __name__ == '__main__':
    char_list, chars = load_goblet_of_fire("../goblet_book.txt")