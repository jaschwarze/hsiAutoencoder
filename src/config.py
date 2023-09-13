FINAL_SELECT_AMOUNT = 30
ALPHA = 0.0001
BETA = 0.1
EPOCHS = 40
BATCH_SIZE = 2500

BAND_SEGMENTS_OK = [
    [*range(0, 45)],
    [*range(45, 80)],
    [*range(80, 133)],
    [*range(133, 184)],
    [*range(184, 224)]
]

BAND_SEGMENTS_GREEN = [
    [*range(0, 69)],
    [*range(69, 104)],
    [*range(104, 142)],
    [*range(142, 193)],
    [*range(193, 224)]
]

BAND_SEGMENTS_DAMAGED = [
    [*range(0, 64)],
    [*range(64, 104)],
    [*range(104, 142)],
    [*range(142, 193)],
    [*range(193, 224)]
]
