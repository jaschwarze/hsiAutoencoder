FINAL_SELECT_AMOUNT = 30
ALPHA = 0.0001
BETA = 0.1
EPOCHS = 40
BATCH_SIZE = 2500

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
