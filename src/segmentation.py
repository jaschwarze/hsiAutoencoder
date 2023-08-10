import matplotlib.pyplot as plt
import numpy
from spectral import *
import spectral.io.envi as envi
import numpy as np


def generate_band_distances(img):
    """
    Generates the absolute difference between two adjacent bands for every band

    :arg:
        img (np array): the hyperspectral image as Numpy-Array

    :return:
        res (np array): Array containing all distances (index i holds the absolute distance from band i to band i+1)
    """

    band_amount = img.shape[2]
    erg = numpy.zeros((band_amount, img.shape[0], img.shape[1]))

    for i in range(0, band_amount - 1):
        r_i = img.read_band(i)
        r_i_p1 = img.read_band(i + 1)
        d_i = abs(r_i_p1 - r_i)
        erg[i] = d_i

    return erg


reduced_image_path = "../preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_9.npy"
output_image_path = "../preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_9_reduced.hdr"

loaded_array = np.load(reduced_image_path)

current_img = envi.create_image(output_image_path,
                        interleave="BIL",
                        dtype="uint16",
                        force=True, ext="hdr",
                        shape=loaded_array.shape)

mm = current_img.open_memmap(writable=True)
mm[:, :, :] = loaded_array

# Generate segments based on correlation matrix
segment_0 = current_img.read_bands([item for item in range(0, 9)])
segment_1 = current_img.read_bands([item for item in range(9, 40)])
segment_2 = current_img.read_bands([item for item in range(40, 150)])
segment_3 = current_img.read_bands([item for item in range(150, 190)])
segment_4 = current_img.read_bands([item for item in range(190, 223)])

d = generate_band_distances(current_img)
s = segment_2.shape[2]


temp_sum = 0
for i in range(1, s-1):
    temp_sum += d[i]

dd_1 = (1/s) * temp_sum

print(dd_1)

