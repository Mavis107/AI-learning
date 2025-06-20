"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, image):
        assert type(image) == np.ndarray, "The input image needs to be a numpy array! Make sure you dont send the string path."
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          
        OldRange = (self._data_max - self._data_min)  
        NewRange = (self.max - self.min)  
        ret_image = (((image - self._data_min) * NewRange) / OldRange) + self.min
        
        return ret_image
    

def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape (C,). 
    """

    # Compute the mean and std along the spatial and batch dimensions (N, H, W)
    mean = np.mean(images, axis=(0,1,2)) # Shape (C,)
    std = np.std(images, axis=(0,1,2)) # Shape (C,)

    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation       
        images = (images - self.mean)/self.std

        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images


class IdentityTransform:
    """Transform class that does nothing"""
    def __call__(self, images):
        return images