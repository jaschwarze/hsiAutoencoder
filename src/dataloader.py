import os
import fnmatch
import numpy as np
import tensorflow as tf


def load_training_data(base_path):
    """
    This functions loads all hyperspectral images from a given location and returns the normed spectra [0, 1] for every
    pixel
    :param base_path: path to the directory containing the numpy-files for the hyperspectral images
    :return: numpy-array with the normed pixel spectra's for every pixel in every image
    """
    if not os.path.exists(base_path):
        raise ValueError("base dir not found")

    image_paths = []
    res = []

    for root, dirs, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, "*.npy"):
            image_paths.append(os.path.join(root, filename))

    for idx, file_path in enumerate(image_paths):
        image_numpy = np.load(file_path)

        if np.ndim(image_numpy) != 3:
            continue

        # mask all black pixels from the pre-processing (reflectance spectra is 0)
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0] * image_numpy.shape[1], image_numpy.shape[2]))
        reflectance_mask = np.sum(image_numpy, axis=1) == 0

        image_numpy = image_numpy[~reflectance_mask]

        for i in range(image_numpy.shape[0]):
            res.append(image_numpy[i])

    res = np.array(res)

    normed_res = tf.math.divide(
        tf.math.subtract(
            res,
            tf.reduce_min(res)
        ),
        tf.math.subtract(
            tf.reduce_max(res),
            tf.reduce_min(res)
        )
    )

    return normed_res.numpy()
