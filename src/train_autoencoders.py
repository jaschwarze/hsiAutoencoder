from dataloader import load_training_data
from config import *
from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder
import os


def train_model(quality, sensor_string="FX10"):
    base_path = os.path.join("../preprocessing/output", sensor_string, quality)
    if not os.path.exists(base_path):
        raise Exception("training data path not found")

    training_data = load_training_data(base_path)

    if quality == "OK":
        current_segments = BAND_SEGMENTS_OK
    elif quality == "Gruen":
        current_segments = BAND_SEGMENTS_GREEN
    elif quality == "Beschaedigt":
        current_segments = BAND_SEGMENTS_DAMAGED
    else:
        raise Exception("image quality not valid")

    segment_aes = []

    for index, segment in enumerate(current_segments):
        img_segment = training_data[:, segment[0]:segment[-1] + 1]
        x_train, x_test = train_test_split(img_segment, test_size=0.4, random_state=1, shuffle=True)

        input_d = x_train.shape[1]
        encoding_d = int(input_d / 2)
        segment_ae = Autoencoder(input_d, encoding_d)
        segment_aes.insert(index, {"id": index, "x_train": x_train, "x_test": x_test,
                                   "ae_object": segment_ae,
                                   "segment_range": str(segment[0]) + ":" + str(segment[-1]),
                                   "training_history": None})

    for config in segment_aes:
        config["ae_object"].compile()
        config["training_history"] = config["ae_object"].train(config["x_train"],
                                                               epochs=40,
                                                               batch_size=2500,
                                                               shuffle=True,
                                                               validation_data=(config["x_test"], config["x_test"]))

        if not os.path.exists("../trained_models"):
            os.mkdir("../trained_models")

        if not os.path.exists("../trained_models/" + sensor_string):
            os.mkdir("../trained_models/" + sensor_string)

        if not os.path.exists("../trained_models/" + sensor_string + "/" + quality):
            os.mkdir("../trained_models/" + sensor_string + "/" + quality)

        ae_save_path = "../trained_models/" + sensor_string + "/" + quality + "/ae_" + quality + "_" + config[
            "segment_range"] + ".keras"

        config["ae_object"].save(ae_save_path)

    return segment_aes


if __name__ == "__main__":
    qualities = ["OK", "Gruen", "Beschaedigt"]

    for quality in qualities:
        train_model(quality, "FX10")
