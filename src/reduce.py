import numpy as np
import fnmatch
import keras
import os
import multiprocessing as mp
from itertools import repeat
import src.config as config


def select_bands(img_path, current_segments, metric_path, final_select_amount, ae_weights):
    if not os.path.isfile(img_path):
        raise ValueError(f"file {img_path} not found")

    dd_file_path = metric_path + "/DD_" + str(final_select_amount) + "BANDS_" + os.path.basename(img_path)
    if not os.path.isfile(dd_file_path):
        raise ValueError(f"distance density file {dd_file_path} not found")

    distance_densities = np.load(dd_file_path)

    erg_bands = []
    for index, normalized_weights in ae_weights.items():
        for band in np.argsort(-normalized_weights)[:int(distance_densities[index])]:
            erg_bands.append(band + current_segments[index][0])

    erg_bands.sort()
    return erg_bands


def get_reduced_image(img_path, current_segments, metric_path, final_select_amount, ae_weights):
    if not os.path.isfile(img_path):
        raise ValueError(f"file {img_path} not found")

    selected_bands = select_bands(img_path, current_segments, metric_path, final_select_amount, ae_weights)
    if len(selected_bands) != final_select_amount:
        raise ValueError(f"selected band amount for file {img_path} not valid")

    img_values = np.load(img_path)
    return np.stack([img_values[:, :, idx] for idx in selected_bands], axis=-1)


def calc_reduced_file(image_path, output_path, current_segments, metric_path, final_select_amount, ae_weights):
    raw_cat_dir = os.path.dirname(os.path.abspath(image_path))
    fig_title = image_path.replace(raw_cat_dir + "/", "")
    cur_out_dir = image_path.replace(fig_title, "").replace(raw_cat_dir, output_path)
    final_image_path = os.path.join(cur_out_dir, "Reduced_" + str(final_select_amount) + "BANDS_" + fig_title)

    if os.path.isfile(final_image_path):
        return

    if not os.path.isdir(cur_out_dir):
        os.makedirs(cur_out_dir, exist_ok=True)

    reduced_img = get_reduced_image(
        image_path,
        current_segments,
        metric_path,
        final_select_amount,
        ae_weights
    )

    np.save(final_image_path, reduced_img)


def reduce_all_images(input_path, output_path, current_segments, metric_path, final_select_amount, ae_weights):
    image_paths = []
    for root, _, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, "*.npy"):
            image_file_path = os.path.join(root, filename)
            image_paths.append(image_file_path)

    with mp.Pool() as pool:
        pool.starmap(
            calc_reduced_file,
            zip(
                image_paths,
                repeat(output_path),
                repeat(current_segments),
                repeat(metric_path),
                repeat(final_select_amount),
                repeat(ae_weights)
            )
        )

if __name__ == "__main__":
    result_dir = config.REDUCED_IMAGE_PATH
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    for segment in config.BAND_SEGMENTS:
        if len(segment) % 2 != 0:
            raise ValueError("Number of segments must be even")

    for category in config.CATEGORIES:
        raw_cat_dir = os.path.join(config.PREPROCESSING_PATH, "output", category)
        if not os.path.exists(raw_cat_dir):
            raise NotADirectoryError(f"masked data directory {raw_cat_dir} does not exist")

        metric_cat_dir = os.path.join(config.METRIC_PATH, category)
        if not os.path.exists(metric_cat_dir):
            raise NotADirectoryError(f"metric data directory {metric_cat_dir} does not exist")

        ae_dir = config.TRAINED_MODEL_PATH
        if not os.path.exists(ae_dir):
            raise NotADirectoryError(f"autoenoder train directory {ae_dir} does not exist")

        output_path = os.path.join(result_dir, category)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ae_weights = dict()
        for index, segment in enumerate(config.BAND_SEGMENTS):
            enc_identifier = (
                "enc_" + str(config.FINAL_SELECT_AMOUNT) + "BANDS_" + str(segment[0]) + ":" + str(segment[-1]) + ".keras"
            )

            enc_path = os.path.join(ae_dir, enc_identifier)
            if not os.path.isfile(enc_path):
                raise Exception(f"trained model file {enc_path} not found")

            current_enc = keras.saving.load_model(enc_path, compile=False)
            hidden_layer = current_enc.get_layer("hidden_layer")
            weights = hidden_layer.get_weights()[0]
            normalized_weights = np.linalg.norm(weights, axis=1, ord=2)
            ae_weights[index] = normalized_weights

        reduce_all_images(
            input_path=raw_cat_dir,
            output_path=output_path,
            current_segments=config.BAND_SEGMENTS,
            metric_path=metric_cat_dir,
            final_select_amount=config.FINAL_SELECT_AMOUNT,
            ae_weights=ae_weights
        )
