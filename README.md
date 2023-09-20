# Segmented Autoencoder for Hyperspectral Bandselection

This repository provides functions to perform hyperspectral bandselection using Segmented Autoencoders.

[Reference paper](https://ieeexplore.ieee.org/document/8611643)


## Usage
1. perform preprocessing using [src/preprocessing.py](src/preprocessing.py)
2. calculate distance density using [src/segmentation.py](src/segmentation.py)
3. train segmented autoencoders with [src/train_autoencoders.py](src/train_autoencoders.py)
4. generate reduced images using [src/band_selection.py](src/band_selection.py)