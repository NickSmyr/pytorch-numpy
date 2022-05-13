import unittest
import numpy as np

from framework.non_pure import batch_chars_dataset


class MyTestCase(unittest.TestCase):
    def test_batch_character(self):
        dataset = 'abcd' * 22
        seq_length = 7
        batch_size = 2
        X_batches, Y_batches = batch_chars_dataset(dataset, batch_size,
                                                seq_length)
        for batch in X_batches[:-1]:
            self.assertTrue(batch.shape == (seq_length,batch_size))

        for batch in Y_batches[:-1]:
            self.assertTrue(batch.shape == (seq_length,batch_size))

if __name__ == '__main__':
    unittest.main()
