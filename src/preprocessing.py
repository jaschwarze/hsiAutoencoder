from skimage.filters import threshold_li
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
from skimage.measure import regionprops
import spectral.io.envi as envi
import os
import numpy as np
import fnmatch
from sklearn.preprocessing import normalize


def calc_hist(reshaped):
    histograms = []
    for i in range(reshaped.shape[-1]):
        histograms.append(np.histogram(reshaped[:, i], bins=100))

    return histograms


def get_channel_max_var(histograms):
    variance = np.zeros(shape=224, dtype=np.float32)
    stds = np.zeros(shape=224, dtype=np.float32)
    for idx, h in enumerate(histograms):
        variance[idx] = np.var(h[1])
        stds[idx] = np.std(h[1])

    return np.argmax(variance)


def prepare_hyper(x_input, masking, x_begin, x_stop):
    cropped_hyper_img = np.zeros(shape=(x_input.shape[0], 768, x_input.shape[2]))
    reduced_hyper_img = np.zeros(shape=(masking.shape[0], masking.shape[1], x_input.shape[2]))

    mask_inverse = np.ones(masking.shape, dtype=np.int8)
    mask_inverse[masking == 1] = 0

    for i in range(x_input.shape[-1]):
        cropped_hyper_img[:, :, i] = x_input[:, 128:896, i]

    for i in range(cropped_hyper_img.shape[-1]):
        tmp_mask = cropped_hyper_img[:, x_begin:x_stop, i]
        tmp_mask[mask_inverse == 1] = 0

        reduced_hyper_img[:, :, i] = tmp_mask

    return reduced_hyper_img


def preprocess():
    try:
        input_path = "preprocessing/input"
        output_path = "preprocessing/output"

        dirname = os.path.dirname(__file__).replace("src", "")
        input_path = os.path.join(dirname, input_path)
        output_path = os.path.join(dirname, output_path)

        if not os.path.exists(input_path):
            raise Exception("input path not found")

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        for root, dirs, filenames in os.walk(input_path):
            for filename in fnmatch.filter(filenames, "*.hdr"):
                hdr_file_path = os.path.join(root, filename)
                bin_file_path = hdr_file_path.replace("hdr", "bin")

                if not os.path.isfile(bin_file_path):
                    print(f".bin file for {hdr_file_path} not found")
                    continue

                ref_file = envi.open(hdr_file_path, bin_file_path)
                x = np.array(ref_file.load())

                x_reshaped = x.reshape(-1, x.shape[2])
                x_normed = x_reshaped / x_reshaped.max(axis=0)

                hist = calc_hist(x_normed)
                channel = get_channel_max_var(hist)

                grey_image = x[:, :, channel]
                grey_image = normalize(grey_image)

                cropped_image = grey_image[:, 128:896]
                thresh_img = threshold_li(cropped_image)
                threshed = cropped_image > thresh_img

                morphed = dilation(threshed, disk(6))
                morphed = erosion(morphed, disk(8))

                binary_img = np.ones(shape=threshed.shape, dtype=np.int8)
                binary_img[morphed] = 0

                regions = regionprops(binary_img)

                final_shape = (600, 600)

                for prop in regions:
                    bbox = prop.bbox

                    print(f"bbox {bbox}")

                    x_diff = final_shape[1] - (bbox[3] - bbox[1])

                    x_diff_1 = int(np.ceil(x_diff / 2))
                    x_diff_2 = int(np.floor(x_diff / 2))

                    mask = np.zeros(shape=binary_img.shape, dtype=np.int8)

                    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = prop.image_filled

                    x_start = bbox[1] - x_diff_1
                    x_end = bbox[3] + x_diff_2
                    mask_reduced = mask[:, x_start:x_end]

                    fig_title = filename.replace(input_path + "/", "")

                    print(f'image title {fig_title}')
                    x_prepared = prepare_hyper(x, mask_reduced, x_start, x_end)

                    cur_out_dir = hdr_file_path.replace(fig_title, "").replace("input", "output")
                    if not os.path.isdir(cur_out_dir):
                        os.makedirs(cur_out_dir)

                    print(hdr_file_path.replace("input", "output").replace("hdr", "npy"))
                    np.save(hdr_file_path.replace("input", "output").replace("hdr", "npy"), x_prepared)

    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    preprocess()
