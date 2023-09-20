import keras
import numpy as np
import tensorflow as tf
import src.config as config
import fnmatch
import os


def select_bands(img_path):
    if not os.path.isfile(img_path):
        raise ValueError("file " + img_path + " not found")

    if "OK" in img_path:
        current_segments = config.BAND_SEGMENTS_OK
        quality = "OK"
    elif "Gruen" in img_path:
        current_segments = config.BAND_SEGMENTS_GREEN
        quality = "Gruen"
    elif "Beschaedigt" in img_path:
        current_segments = config.BAND_SEGMENTS_DAMAGED
        quality = "Beschaedigt"
    else:
        raise ValueError("image quality not found")

    if "FX10" in img_path:
        sensor_string = "FX10"
    elif "FX10" in img_path:
        sensor_string = "FX17"
    else:
        raise ValueError("sensor type not found")

    dd_file_path = os.path.join(config.METRIC_PATH, sensor_string, quality) + "/DD_" + os.path.basename(img_path)
    if not os.path.isfile(dd_file_path):
        raise ValueError("distance density file " + dd_file_path + " not found")

    distance_densities = np.load(dd_file_path)
    models_path = os.path.join(config.TRAINED_MODEL_PATH, sensor_string, quality)

    if not os.path.exists(models_path):
        raise ValueError("path to trained models not found")

    erg_bands = []

    for index, segment in enumerate(current_segments):
        enc_identifier = "enc_" + quality + "_" + str(segment[0]) + ":" + str(segment[-1]) + ".keras"
        enc_path = os.path.join(models_path, enc_identifier)

        if not os.path.isfile(enc_path):
            raise Exception("trained model file " + enc_path + " not found")

        current_enc = keras.models.load_model(enc_path, compile=False)
        hidden_layer = current_enc.get_layer("hidden_layer")
        weights = hidden_layer.get_weights()[0]
        normalized_weights = tf.norm(weights, axis=1, ord=2)
        normalized_weights.numpy()

        for band in np.argsort(-normalized_weights)[:int(distance_densities[index])]:
            erg_bands.append(band + current_segments[index][0])

    erg_bands.sort()

    return erg_bands


def get_reduced_image(img_path):
    if not os.path.isfile(img_path):
        raise ValueError("file " + img_path + " not found")

    selected_bands = select_bands(img_path)
    img_values = np.load(img_path)

    return np.stack([img_values[:, :, idx] for idx in selected_bands], axis=-1)


def reduce_all_images():
    input_path = os.path.join(config.PREPROCESSING_PATH, "output")
    output_path = config.REDUCED_IMAGE_PATH

    if not os.path.exists(input_path):
        raise Exception("input path not found")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for root, dirs, filenames in os.walk(input_path):
        for filename in fnmatch.filter(filenames, "*.npy"):
            image_file_path = os.path.join(root, filename)

            reduced_img = get_reduced_image(image_file_path)
            fig_title = filename.replace(input_path + "/", "")

            cur_out_dir = image_file_path.replace(fig_title, "").replace(input_path, output_path[:-1])
            if not os.path.isdir(cur_out_dir):
                os.makedirs(cur_out_dir)

            final_image_path = os.path.join(cur_out_dir, "Reduced_" + fig_title)
            np.save(final_image_path, reduced_img)


if __name__ == "__main__":
    reduce_all_images()
