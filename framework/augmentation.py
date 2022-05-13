from itertools import product
from typing import List, Tuple

import numpy as np
import random

class TransformOnFlat:
    """Transform applied to flat image"""
    def apply(self, x):
        raise NotImplementedError

class FlipImage(TransformOnFlat):

    def __init__(self):
        pass

    def apply(self, x):
        numimages = x.shape[3]
        return np.flip(x, axis=2)

class UnFlattenImage(TransformOnFlat):
    def __init__(self, image_width, image_height, num_channels):
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels

    def apply(self, x):
        numimages = x.shape[1]
        return x.reshape(self.num_channels,self.image_width,
                         self.image_height,
                         numimages)

class FlattenImage(TransformOnFlat):
    def __init__(self, image_width, image_height, num_channels):
        self.image_width = image_width
        self.image_height = image_height
        self.num_channels = num_channels

    def apply(self, x):
        numimages = x.shape[3]
        return x.reshape(self.num_channels*self.image_width*
                         self.image_height,
                         numimages)

class TranslateImage(TransformOnFlat):
    def __init__(self, tx, ty):
        self.tx = tx
        self.ty = ty

    def apply(self, x):
        tx = self.tx
        ty = self.ty
        numimages = x.shape[3]
        visible_image = x

        if tx < 0:
            x_range = slice(0, 32 + tx)
        else:
            x_range = slice(tx, 32)

        if ty < 0:
            y_range = slice(0, 32 + ty)
        else:
            y_range = slice(ty, 32)

        visible_image = np.roll(visible_image, ty, axis=1)
        visible_image = np.roll(visible_image, tx, axis=2)
        ret_img = np.zeros_like(visible_image)
        ret_img[:, y_range, x_range, :] = visible_image[:, y_range, x_range, :]
        return ret_img

# TODO
# class EqualPartsAugmenter(TransformOnFlat):
#     """
#     Augmenter that splits the input batch into as many equal parts as input transformations
#     and applies each input to each of them
#     """
#     def __init__(self, transform_list : List[TransformOnFlat]):
#         self.transform_list = transform_list
#
#     def apply(self, x):
#         for x in self.transform_list:
#             batch_size = x.shape[1]
#             idxs_to_flip = random.sample(range(batch_size), k=int(percentage * batch_size))
#             x[:, idxs_to_flip] = transform.apply(x[:, idxs_to_flip])

class BatchAugmenter(TransformOnFlat):
    """
    Class responsible for data augmentation within a batch. Applies each given transform
    to a random subset of the given (flat) images
    """
    def __init__(self, transform_list : List[Tuple[TransformOnFlat, float]]):
        """
        :param transform_list: A list of transforms to apply as well as their percentages
        """
        self.transform_list = transform_list

    def apply(self, x):
        """
        x: Batch of shape (dim, n_batch)
        """
        # Copy array because we need to mutate it
        x = np.copy(x)
        for transform, percentage in self.transform_list:
            # Operations that modify the structure of the np array must only be
            # with percentage 1
            if percentage == 1:
                x = transform.apply(x)
            else:
                batch_size = x.shape[3]
                idxs_to_flip = random.sample(range(batch_size), k=int(percentage * batch_size))
                x[..., idxs_to_flip] = transform.apply(x[..., idxs_to_flip])

        return x

class BatchExtenderFlatImages(TransformOnFlat):
    """
    Creates new samples based on the transforms and returns the old and new
    samples
    """
    def __init__(self, transform_list : List[Tuple[TransformOnFlat, float]]):
        """
        :param transform_list: A list of transforms to apply as well as their percentages
        """
        self.transform_list = transform_list

    def apply(self, x):
        """
        x: Batch of shape (dim, n_batch)
        """
        # Copy array because we need to mutate it
        x = np.copy(x)
        x = UnFlattenImage(32, 32, 3).apply(x)
        augmented_xs = [x]
        for transform, percentage in self.transform_list:
            # Operations that modify the structure of the np array must only be
            batch_size = x.shape[3]
            idxs_to_flip = random.sample(range(batch_size),
                                         k=int(percentage * batch_size))

            new_augmented = transform.apply(x[..., idxs_to_flip])
            augmented_xs.append(new_augmented)

        flatten = FlattenImage(32, 32 , 3)
        augmented_xs = [flatten.apply(x) for x in augmented_xs]
        return np.concatenate(augmented_xs, axis=1)

class Assignment2Augmenter(BatchAugmenter):
    def __init__(self, pflip, ptrans, num_channels, image_width,
                 image_height):
        super(Assignment2Augmenter, self).__init__([
            (UnFlattenImage(image_width,image_height,num_channels),
             1),
            (FlipImage(), pflip),
            *[(TranslateImage(tx,ty), ptrans/4) for tx,ty in product([-3,3], [-3,3])],
            (FlattenImage(image_width,image_height,num_channels), 1)
        ])