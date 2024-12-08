import numpy as np
import os
import fnmatch
import logging
import multiprocessing as mp
from itertools import repeat
import src.config as config


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

    nb_sum = np.sum(nb)
    difference = final_select_amount - nb_sum

    if difference > 0:
        min_index = np.argmin(nb)
        nb[min_index] += difference
    elif difference < 0:
        max_index = np.argmax(nb)

        if (nb[max_index] + difference) > 0:
            nb[max_index] += difference

    for idx, seg in enumerate(predefined_segments):
        seg_len = seg[1] - seg[0]

        if nb[idx] > seg_len:
            diff = nb[idx] - seg_len
            min_index = np.argmin(nb)

            if (nb[idx] - diff) > 0:
                nb[idx] -= diff
                nb[min_index] += diff

    for idx, seg in enumerate(predefined_segments):
        seg_len = seg[1] - seg[0]
        if nb[idx] > seg_len or nb[idx] < 0:
            raise ValueError(f"Invalid band amount")

    if np.sum(nb) != final_select_amount:
        raise ValueError(f"Invalid band amount")

    return nb


def calc_metric_file(image_path, output_path, current_segments, final_select_amount):
    raw_cat_dir = os.path.dirname(os.path.abspath(image_path))
    fig_title = image_path.replace(raw_cat_dir + "/", "")
    cur_out_dir = image_path.replace(fig_title, "").replace(raw_cat_dir, output_path)

    final_image_path = os.path.join(cur_out_dir, "DD_" + str(final_select_amount) + "BANDS_" + fig_title)
    if os.path.isfile(final_image_path):
        return

    loaded_image = np.load(image_path)
    bands_per_segment = get_distance_density(loaded_image, current_segments, final_select_amount)

    if not os.path.isdir(cur_out_dir):
        os.makedirs(cur_out_dir)

    logging.info("Final im path:", final_image_path)
    np.save(final_image_path, bands_per_segment)


def calc_segment(input_path, output_path, current_segments, final_select_amount):
    image_paths = []
    for root, _, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, "*.npy"):
            image_file_path = os.path.join(root, filename)
            image_paths.append(image_file_path)

    with mp.Pool() as pool:
        pool.starmap(
            calc_metric_file,
            zip(
                image_paths,
                repeat(output_path),
                repeat(current_segments),
                repeat(final_select_amount)
            )
        )

if __name__ == "__main__":
    try:
        input_path = os.path.join(config.PREPROCESSING_PATH, "output")
        output_path = config.METRIC_PATH

        if not os.path.exists(input_path):
            raise NotADirectoryError("input path not found")

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        for segment in config.BAND_SEGMENTS:
            if len(segment) % 2 != 0:
                raise ValueError("Number of segments must be even")

        for category in config.CATEGORIES:
            raw_cat_dir = os.path.join(input_path, category)
            if not os.path.exists(raw_cat_dir):
                raise NotADirectoryError(f"masked data directory {raw_cat_dir} does not exist")

            metric_cat_dir = os.path.join(output_path, category)
            if not os.path.exists(metric_cat_dir):
                os.makedirs(metric_cat_dir, exist_ok=True)

            calc_segment(raw_cat_dir, metric_cat_dir, config.BAND_SEGMENTS, config.FINAL_SELECT_AMOUNT)
    except Exception as e:
        print(str(e))
