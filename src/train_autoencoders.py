import math
import multiprocessing
import os
import tensorflow as tf
from autoencoder import Autoencoder
from dataloader import HyperspectralSegmentLoader
import src.config as config


if __name__ == "__main__":
    model_dir = config.TRAINED_MODEL_PATH
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    for segment in config.BAND_SEGMENTS:
        if len(segment) % 2 != 0:
            raise ValueError("Number of segments must be even")

    input_train_dir = os.path.join(config.NORMALIZED_PATH, "train")
    if not os.path.exists(input_train_dir):
        raise NotADirectoryError(f"normalized data directory {input_train_dir} for train data does not exist")

    input_test_dir = os.path.join(config.NORMALIZED_PATH, "test")
    if not os.path.exists(input_test_dir):
        raise NotADirectoryError(f"normalized data directory {input_test_dir} for test data does not exist")

    segment_aes = []
    bands = 0
    for index, segment in enumerate(config.BAND_SEGMENTS):
        ae_save_path = model_dir + "/enc_" + str(config.FINAL_SELECT_AMOUNT) + "BANDS_" + str(segment[0]) + ":" + str(segment[1]) + ".keras"
        if os.path.exists(ae_save_path):
            continue

        input_dim = segment[1] - segment[0]
        encoding_dim = math.ceil(input_dim / 2)

        with tf.device("/GPU:0"):
            segment_ae = Autoencoder(input_dim, encoding_dim, config.ALPHA, config.BETA)

        train_loader = HyperspectralSegmentLoader(
            workers=int(multiprocessing.cpu_count() // 2),
            use_multiprocessing=True,
            base_dir=input_train_dir,
            batch_size=config.BATCH_SIZE,
            segment_start=segment[0],
            segment_end=segment[1]
        )

        test_loader = HyperspectralSegmentLoader(
            workers=int(multiprocessing.cpu_count() // 2),
            use_multiprocessing=True,
            base_dir=input_test_dir,
            batch_size=config.BATCH_SIZE,
            segment_start=segment[0],
            segment_end=segment[1]
        )

        segment_aes.insert(index, {
            "id": index,
            "ae_object": segment_ae,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "segment_range": str(segment[0]) + ":" + str(segment[1]),
            "training_history": None,
            "save_path": ae_save_path
        })

    for ae_config in segment_aes:
        ae_config["training_history"] = ae_config["ae_object"].train(
            data=ae_config["train_loader"],
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=ae_config["test_loader"],
            shuffle=False
        )

        ae_config["ae_object"].save_encoder(ae_config["save_path"])
