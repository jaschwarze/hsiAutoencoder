import numpy as np
import os
import fnmatch
from src.config import *


def get_distance_density(img_array, predefined_segments, final_select_amount):
    """ Calculates the distance density for a pre-segmented hyperspectral image
    :param img_array: the hyperspectral image as numpy-array
    :param predefined_segments: a list of pre-defined band subregions
    :param final_select_amount: the amount of bands for the final selection
    :return: array containing all distance densities for the given segments
    """
    if np.ndim(img_array) != 3:
        raise ValueError("the input image should have 3 dimensions")

    image_band_amount = img_array.shape[2]

    r = np.zeros(image_band_amount)
    d = np.zeros(image_band_amount)
    dd = np.zeros(len(predefined_segments))
    nb = np.zeros(len(predefined_segments))

    for k in range(0, image_band_amount):
        r[k] = np.sum(img_array[:, :, k])

    for i in range(0, image_band_amount - 1):
        d[i] = np.abs(np.subtract(r[i + 1], r[i]))

    for segment_index, segment in enumerate(predefined_segments):
        seg_start = segment[0]
        n = len(segment)

        for i in range(seg_start, seg_start + n):
            dd[segment_index] += d[i]

        dd[segment_index] *= (1 / n)

    for i in range(len(predefined_segments)):
        nb[i] = np.round((dd[i] / np.sum(dd)) * final_select_amount)

    return nb


def segment():
    try:
        input_path = "preprocessing/output"
        output_path = "segmentation/"

        dirname = os.path.dirname(__file__).replace("src", "")
        input_path = os.path.join(dirname, input_path)
        output_path = os.path.join(dirname, output_path)

        if not os.path.exists(input_path):
            raise Exception("input path not found")

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for root, dirs, filenames in os.walk(input_path):
            for filename in fnmatch.filter(filenames, "*.npy"):
                image_file_path = os.path.join(root, filename)
                loaded_image = np.load(image_file_path)

                if "OK" in image_file_path:
                    current_segments = BAND_SEGMENTS_OK
                elif "Gruen" in image_file_path:
                    current_segments = BAND_SEGMENTS_GREEN
                elif "Beschaedigt" in image_file_path:
                    current_segments = BAND_SEGMENTS_DAMAGED
                else:
                    raise Exception("image quality not found")

                bands_per_segment = get_distance_density(loaded_image, current_segments, FINAL_SELECT_AMOUNT)

                fig_title = filename.replace(input_path + "/", "")
                cur_out_dir = image_file_path.replace(fig_title, "").replace(input_path, output_path[:-1])
                if not os.path.isdir(cur_out_dir):
                    os.makedirs(cur_out_dir)

                final_image_path = os.path.join(cur_out_dir, "DD_" + fig_title)
                np.save(final_image_path, bands_per_segment)

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    segment()
