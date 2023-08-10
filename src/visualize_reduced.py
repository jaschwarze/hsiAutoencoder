from spectral import *
import spectral.io.envi as envi
import numpy as np

normale_image_path = "./preprocessing/input/FX10/Gruen/Kartoffel_Fontane_FX10_8.hdr"
normale_image_path_bin = "./preprocessing/input/FX10/Gruen/Kartoffel_Fontane_FX10_8.bin"

reduced_image_path = "./preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_8.npy"
output_image_path = "./preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_8_reduced.hdr"

loaded_array = np.load(reduced_image_path)

img = envi.create_image(output_image_path,
                        interleave="BIL",
                        dtype="uint16",
                        force=True, ext="hdr",
                        shape=loaded_array.shape)

mm = img.open_memmap(writable=True)

mm[:, :, :] = loaded_array


view = imshow(img, title="Ohne Fließband", bands=(160,))

old_img = envi.open(normale_image_path, normale_image_path_bin)
old_view = imshow(old_img, title="Original")

pc = principal_components(old_img)
v = imshow(pc.cov)
