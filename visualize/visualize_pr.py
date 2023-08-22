from spectral import *
import spectral.io.envi as envi
import numpy as np

# Grün, OK → Kartoffel 9
# Beschädigt → Kartoffel 8

# OK
ok_image_path = "./preprocessing/output/FX10/OK/Kartoffel_Premiere_FX10_9.npy"
ok_output_image_path = "./preprocessing/output/FX10/OK/Kartoffel_Premiere_FX10_9_reduced.hdr"
ok_loaded_array = np.load(ok_image_path)
ok_img = envi.create_image(ok_output_image_path, interleave="BIL", dtype="uint16", force=True, ext="hdr", shape=ok_loaded_array.shape)
ok_mm = ok_img.open_memmap(writable=True)
ok_mm[:, :, :] = np.fliplr(ok_loaded_array)
ok_view = imshow(ok_img, title="OK", bands=(112, ))
ok_pc = principal_components(ok_img)
ok_ok = imshow(ok_pc.cov, title="Korrelation OK")

# Grün
green_image_path = "./preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_9.npy"
green_output_image_path = "./preprocessing/output/FX10/Gruen/Kartoffel_Fontane_FX10_9_reduced.hdr"
green_loaded_array = np.load(green_image_path)
green_img = envi.create_image(green_output_image_path, interleave="BIL", dtype="uint16", force=True, ext="hdr", shape=green_loaded_array.shape)
green_mm = green_img.open_memmap(writable=True)
green_mm[:, :, :] = np.fliplr(green_loaded_array)
green_view = imshow(green_img, title="Grün", bands=(112, ))
green_pc = principal_components(green_img)
green_v = imshow(green_pc.cov, title="Korrelation Grün")

# Beschädigt
bs_image_path = "./preprocessing/output/FX10/Beschaedigt/Kartoffel_Premiere_FX10_8.npy"
bs_output_image_path = "./preprocessing/output/FX10/Beschaedigt/Kartoffel_Premiere_FX10_8_reduced.hdr"
bs_loaded_array = np.load(bs_image_path)
bs_img = envi.create_image(bs_output_image_path, interleave="BIL", dtype="uint16", force=True, ext="hdr", shape=bs_loaded_array.shape)
bs_mm = bs_img.open_memmap(writable=True)
bs_mm[:, :, :] = np.fliplr(bs_loaded_array)
bs_view = imshow(bs_img, title="Beschädigt", bands=(112, ))
bs_pc = principal_components(bs_img)
bs_v = imshow(bs_pc.cov, title="Korrelation Beschädigt")
