"""Shared constants. Frozen interface: see docs/architecture.md before changing anything here."""

import numpy as np

N_PIX = 1024
LASER_NM = 785.0

GRID_START = 600.0
GRID_STOP = 1800.0
GRID_STEP = 1.0
GRID = np.arange(GRID_START, GRID_STOP + GRID_STEP / 2, GRID_STEP)

TRUNCATE_BELOW_CM1 = 450.0

CLASSES = ("normal", "cancer")

TYLENOL_CAL_PEAKS_CM1 = (797.0, 858.0, 1169.0, 1237.0, 1324.0, 1609.0, 1648.0)
