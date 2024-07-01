import os
from constants import ROOT_PATH


def _embed_js_contents(filename):
    js_path = os.path.join(ROOT_PATH, "modules", "callbacks", filename)

    with open(js_path) as f:
        return f.read()


CALLBACKS = {}

CALLBACKS["som_cb"] = _embed_js_contents("som_cb.js")

CALLBACKS["geo_cb"] = _embed_js_contents("geo_cb.js")
