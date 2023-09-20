from sklearn.model_selection import train_test_split
import os
import src.config as config
from src.autoencoder import Autoencoder
from src.dataloader import load_training_data


def train_model(quality, sensor_string="FX10"):
    base_path = os.path.join(config.PREPROCESSING_PATH, "output", sensor_string, quality)
    if not os.path.exists(base_path):
        raise Exception("training data path not found")

    training_data = load_training_data(base_path)

    if quality == "OK":
        current_segments = config.BAND_SEGMENTS_OK
    elif quality == "Gruen":
        current_segments = config.BAND_SEGMENTS_GREEN
    elif quality == "Beschaedigt":
        current_segments = config.BAND_SEGMENTS_DAMAGED
    else:
        raise Exception("image quality not valid")

    segment_aes = []

    for index, segment in enumerate(current_segments):
        img_segment = training_data[:, segment[0]:segment[-1] + 1]
        x_train, x_test = train_test_split(img_segment, test_size=config.TEST_RATE, random_state=1, shuffle=True)

        input_d = x_train.shape[1]
        encoding_d = int(input_d / 2)
        segment_ae = Autoencoder(input_d, encoding_d)
        segment_aes.insert(index, {"id": index, "x_train": x_train, "x_test": x_test,
                                   "ae_object": segment_ae,
                                   "segment_range": str(segment[0]) + ":" + str(segment[-1]),
                                   "training_history": None})

    for ae_config in segment_aes:
        ae_config["training_history"] = ae_config["ae_object"].train(ae_config["x_train"],
                                                                     epochs=config.EPOCHS,
                                                                     batch_size=config.BATCH_SIZE,
                                                                     shuffle=True,
                                                                     validation_data=(
                                                                     ae_config["x_test"], ae_config["x_test"]))

        output_dir = config.TRAINED_MODEL_PATH
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not os.path.exists(os.path.join(output_dir, sensor_string)):
            os.mkdir(os.path.join(output_dir, sensor_string))

        if not os.path.exists(os.path.join(output_dir, sensor_string, quality)):
            os.mkdir(os.path.join(output_dir, sensor_string, quality))

        ae_save_path = os.path.join(output_dir, sensor_string, quality) + "/enc_" + quality + "_" + ae_config[
            "segment_range"] + ".keras"

        ae_config["ae_object"].save_encoder(ae_save_path)

    return segment_aes


if __name__ == "__main__":
    qualities = ["OK", "Gruen", "Beschaedigt"]

    for quality in qualities:
        train_model(quality, "FX10")
