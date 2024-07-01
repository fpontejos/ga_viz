import os

from pysal.lib import weights as pysal_weights

ROOT_PATH = os.path.dirname(__file__)
# OUT_PATH = os.path.join(ROOT_PATH, os.path.join('internal', 'data', 'out'))

RANDOM_STATE = 1
TOPNN_K = 10

DEFAULT_FEATURES = [
    "TotPop90",
    "PctRural",
    "PctBach",
    "PctEld",
    "PctFB",
    "PctPov",
    "PctBlack",
]
DEFAULT_GEO_FEATS = ["X", "Y"]

DEFAULT_FEATS = {
    "all": DEFAULT_FEATURES,
    "geo": {"x": "X", "y": "Y"},
    "suf": {"scaled": "_std"},
    "x": ["PctFB", "PctBlack", "PctRural"],
    "y": "PctBach",
}


WEIGHTS_MX = {
    "Queen": pysal_weights.Queen,
    "Rook": pysal_weights.Rook,
    "KNN": pysal_weights.KNN,
    "lat2W": pysal_weights.lat2W,
}


PLOT_WIDTH = 400
PLOT_HEIGHT = 700
CONTRAST_COLOR1 = "#ff6361"
CONTRAST_COLOR2 = "#bc5090"

DEFAULT_SOM_PARAMS = {
    "epochs": 30,
    "random_state": 0,
    "learning_rate": 0.7,
    "initialization": "random",
    "m": 9,
    "n": 12,
}
