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

BAND_SEGMENTS = [(0, 27), (27, 53), (53, 105), (105, 148), (148, 173), (173, 224)]
CATEGORIES = ["ok", "gruen", "verformt", "faulig"]