import unittest

from framework.augmentation import Assignment2Augmenter
from framework.non_pure import deterministic_seed, load_cifar


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        deterministic_seed(6)
        cls.trainX, cls.trainY, cls.valX, cls.valY, cls.testX,\
            cls.testY = load_cifar("../../cifar-10-batches-py")

    def test_assignment_2_augmenter_can_be_applied(self):
        augmenter = Assignment2Augmenter(0.5, 0.5, 3, 32 ,32)
        start_idx = 5
        end_idx = 10
        numImages = end_idx - start_idx
        img_shape = (3 * 32 * 32, numImages)

        img = self.trainX[:, start_idx:end_idx].reshape(img_shape)
        augmenter.apply(img)

    def test_assignment_2_training(self):
        pass

if __name__ == '__main__':
    unittest.main()
