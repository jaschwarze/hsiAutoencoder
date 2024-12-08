import math
import os
import fnmatch
import cupy as cp
import numpy as np
import pickle
import src.config as config


def move_files(files, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    for file in files:
        os.rename(file, os.path.join(destination_folder, os.path.basename(file)))


def count_numpy_files(directory_path):
    file_counter = 0
    for _, _, names in os.walk(directory_path):
        for _ in fnmatch.filter(names, "*.npy"):
            file_counter += 1

    return file_counter


if __name__ == "__main__":
    normalized_dir = config.NORMALIZED_PATH
    if not os.path.exists(normalized_dir):
        os.makedirs(normalized_dir, exist_ok=True)

    min_file_path = normalized_dir + "/min.pickle"
    max_file_path = normalized_dir + "/max.pickle"

    if os.path.isfile(min_file_path):
        with open(min_file_path, "rb") as handle:
            dataset_min = pickle.load(handle)
    else:
        dataset_min = math.inf

    if os.path.isfile(max_file_path):
        with open(max_file_path, "rb") as handle:
            dataset_max = pickle.load(handle)
    else:
        dataset_max = -math.inf

    image_paths = []
    new_image_paths = []
    min_max_changed = False

    for category in config.CATEGORIES:
        raw_cat_dir = os.path.join(config.PREPROCESSING_PATH, "output", category)
        if not os.path.exists(raw_cat_dir):
            continue

        for root, dirs, filenames in os.walk(raw_cat_dir):
            for filename in fnmatch.filter(filenames, "*.npy"):
                image_paths.append(os.path.join(root, filename))

    for idx, file_path in enumerate(image_paths):
        normed_file_path = normalized_dir + f"/{os.path.basename(file_path)}"
        normed_train_file_path = normalized_dir + f"/train/{os.path.basename(file_path)}"
        normed_test_file_path = normalized_dir + f"/test/{os.path.basename(file_path)}"

        if os.path.isfile(normed_file_path) or os.path.isfile(normed_train_file_path) or os.path.isfile(normed_test_file_path):
            continue

        new_image_paths.append(normed_file_path)

        image_numpy = cp.load(file_path)
        if cp.ndim(image_numpy) != 3:
            continue

        # remove all black pixels from the pre-processing (reflectance spectra is 0)
        image_numpy = cp.reshape(image_numpy, (image_numpy.shape[0] * image_numpy.shape[1], image_numpy.shape[2]))
        reflectance_mask = cp.sum(image_numpy, axis=1) == 0
        image_numpy = image_numpy[~reflectance_mask]

        arr_min = cp.min(image_numpy)
        arr_max = cp.max(image_numpy)

        if arr_min < dataset_min:
            dataset_min = arr_min
            with open(min_file_path, "wb") as handle:
                pickle.dump(dataset_min, handle)
            min_max_changed = True

        if arr_max > dataset_max:
            dataset_max = arr_max
            with open(max_file_path, "wb") as handle:
                pickle.dump(dataset_max, handle)
            min_max_changed = True

        cp.save(normed_file_path, image_numpy)

    # normalize data to open interval (0, 1)
    # re-normalize all files if min-max have changed, otherwise only new images will be normalized
    if min_max_changed:
        paths_to_normalize = image_paths
    else:
        paths_to_normalize = new_image_paths

    for idx, file_path in enumerate(paths_to_normalize):
        normed_file_path = normalized_dir + f"/{os.path.basename(file_path)}"
        normed_train_file_path = normalized_dir + f"/train/{os.path.basename(file_path)}"
        normed_test_file_path = normalized_dir + f"/test/{os.path.basename(file_path)}"

        if os.path.isfile(normed_train_file_path):
            normed_file_path = normed_train_file_path
        elif os.path.isfile(normed_test_file_path):
            normed_file_path = normed_test_file_path

        image_numpy = cp.load(normed_file_path)
        image_numpy = (image_numpy - dataset_min) / (dataset_max - dataset_min)
        image_numpy = image_numpy * (1 - 2 * config.EPSILON) + config.EPSILON
        cp.save(normed_file_path, image_numpy)

    files_to_split = []
    for root, dirs, filenames in os.walk(normalized_dir):
        for filename in fnmatch.filter(filenames, "*.npy"):
            files_to_split.append(os.path.join(root, filename))

    train_dir = os.path.join(normalized_dir, "train")
    test_dir = os.path.join(normalized_dir, "test")

    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        train_amount = count_numpy_files(train_dir)
        test_amount = count_numpy_files(test_dir)

        train_wanted = len(files_to_split) * config.TRAIN_RATE
        test_wanted = len(files_to_split) - train_wanted

        if train_amount == train_wanted and test_amount == test_wanted:
            quit(0)

    if len(files_to_split) > 0:
        np.random.shuffle(files_to_split)
        split_point = int(len(files_to_split) * config.TRAIN_RATE)
        train_files = files_to_split[:split_point]
        test_files = files_to_split[split_point:]
        move_files(train_files, train_dir)
        move_files(test_files, test_dir)
