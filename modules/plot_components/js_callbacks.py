import os
from constants import ROOT_PATH

def _embed_js_contents(filename):
    js_path = os.path.join(ROOT_PATH, 'modules', 'plot_components', 'js', filename)
    print(js_path)
    with open(js_path) as f:
        return f.read()

CALLBACKS = {}

CALLBACKS['user_callback_code'] = _embed_js_contents('user_callback.js')

CALLBACKS['row_select_cb'] = _embed_js_contents('row_select_callback.js')

CALLBACKS['hex_select_cb'] = _embed_js_contents('hex_select_callback.js')