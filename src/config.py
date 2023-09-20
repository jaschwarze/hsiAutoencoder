import os
ROOT_DIR = os.path.abspath(os.pardir)

PREPROCESSING_PATH = os.path.join(ROOT_DIR, "data/preprocessing/")
METRIC_PATH = os.path.join(ROOT_DIR, "data/segmentation/")
TRAINED_MODEL_PATH = os.path.join(ROOT_DIR, "data/trained_models/")
REDUCED_IMAGE_PATH = os.path.join(ROOT_DIR, "data/reduced_images/")

FINAL_SELECT_AMOUNT = 30
ALPHA = 0.0001
BETA = 0.1
EPOCHS = 1200
BATCH_SIZE = 2500
TEST_RATE = 0.4

BAND_SEGMENTS_OK = [
    [*range(0, 36)],
    [*range(36, 71)],
    [*range(71, 133)],
    [*range(133, 224)]
]

BAND_SEGMENTS_GREEN = [
    [*range(0, 30)],
    [*range(30, 114)],
    [*range(114, 224)]
]

BAND_SEGMENTS_DAMAGED = [
    [*range(0, 28)],
    [*range(28, 64)],
    [*range(64, 132)],
    [*range(132, 224)]
]
