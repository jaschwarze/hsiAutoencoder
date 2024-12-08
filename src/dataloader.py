import keras
import os
import fnmatch
import numpy as np


class HyperspectralSegmentLoader(keras.utils.PyDataset):
    def __init__(self, base_dir, batch_size, segment_start, segment_end, **kwargs):
        super().__init__(**kwargs)
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.image_paths = []
        self.number_samples = 0
        self.file_sample_indices = dict()
        self.segment_start = segment_start
        self.segment_end = segment_end

        for root, dirs, filenames in os.walk(self.base_dir):
            for filename in fnmatch.filter(filenames, "*.npy"):
                self.image_paths.append(os.path.join(root, filename))

        for idx, file_path in enumerate(self.image_paths):
            image_numpy = np.load(file_path)

            if np.ndim(image_numpy) != 2:
                continue

            image_numpy = image_numpy[:, self.segment_start:self.segment_end]

            self.file_sample_indices[file_path] = [self.number_samples, self.number_samples + image_numpy.shape[0] - 1]
            self.number_samples += image_numpy.shape[0]

    def __len__(self):
        return int(np.floor(self.number_samples / self.batch_size))

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, self.number_samples) - 1

        files_to_load = {path: [] for path in self.file_sample_indices}

        for file_path, indices in self.file_sample_indices.items():
            lower_bound = indices[0]
            upper_bound = indices[1]

            if lower_bound <= low <= upper_bound:
                files_to_load[file_path].append(low)
                if high > upper_bound:
                    files_to_load[file_path].append(upper_bound)

            if lower_bound <= high <= upper_bound:
                files_to_load[file_path].append(high)
                if low < lower_bound:
                    files_to_load[file_path].append(lower_bound)

            files_to_load[file_path].sort()

        files_to_load = {k: v for k, v in files_to_load.items() if len(v) == 2}

        return_arrays = []
        for path, indices in files_to_load.items():
            sample_offset = indices[0]

            image_numpy = np.load(path)
            image_numpy = image_numpy[:, self.segment_start:self.segment_end]
            return_arrays.append(image_numpy[indices[0] - sample_offset:indices[1] - sample_offset + 1, :])

        if len(return_arrays):
            x, y = np.concatenate(return_arrays, axis=0), np.concatenate(return_arrays, axis=0)
        else:
            return None

        return x, y
