import keras
import numpy as np
import os
from autoencoder import *
import tensorflow as tf


def select_bands(img_path):
    if not os.path.isfile(img_path):
        raise ValueError("file " + img_path + " not found")

    if "OK" in img_path:
        current_segments = BAND_SEGMENTS_OK
        quality = "OK"
    elif "Gruen" in img_path:
        current_segments = BAND_SEGMENTS_GREEN
        quality = "Gruen"
    elif "Beschaedigt" in img_path:
        current_segments = BAND_SEGMENTS_DAMAGED
        quality = "Beschaedigt"
    else:
        raise ValueError("image quality not found")

    if "FX10" in img_path:
        sensor_string = "FX10"
    elif "FX10" in img_path:
        sensor_string = "FX17"
    else:
        raise ValueError("sensor type not found")

    dd_file_path = "../segmentation/" + sensor_string + "/" + quality + "/" + "DD_" + os.path.basename(img_path)
    if not os.path.isfile(dd_file_path):
        raise ValueError("distance density file " + dd_file_path + " not found")

    distance_densities = np.load(dd_file_path)

    models_path = "../trained_models/" + sensor_string + "/" + quality

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
        normalized_weights = tf.norm(weights, axis=1)
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
