import unittest

from framework.augmentation import TranslateImage
from framework.non_pure import load_cifar
from test.convenience import plot_cifar_image_batch

"""
Tests that can not be automated/ have not been automated yet
"""

class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trainX, cls.trainY, cls.valX, cls.valY, cls.testX, \
        cls.testY = load_cifar(
            "../../cifar-10-batches-py")

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_translate(self):
        img_shape = (3, 32, 32, 1)
        img = self.trainX[:, 2:3].reshape(img_shape)
        trans = TranslateImage(3, -3)
        applied = trans.apply(img)
        plot_cifar_image_batch(img)
        plot_cifar_image_batch(applied)


if __name__ == '__main__':
    unittest.main()


