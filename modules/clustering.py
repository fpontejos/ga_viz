import pickle

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from constants import *
from minisom import MiniSom

som_params = DEFAULT_SOM_PARAMS


def get_som(df, features, som_params=som_params, use_pretrained=False):

    mn = (som_params["m"], som_params["n"])

    if use_pretrained == True:
        som_gp = get_pretrained_som()
        # TODO: make pickle-able
        pass

    p_ = len(features)
    features_scaled = ["{}{}".format(i, "_std") for i in features]

    def inverse_decay_to_zero(learning_rate, t, max_iter):
        """Decay function of the learning process that asymptotically
        approaches zero.
        """
        C = max_iter / 100.0
        return learning_rate * C / (C + t)

    def inverse_decay_to_one(learning_rate, t, max_iter):
        """Decay function of sigma that asymptotically approaches one."""
        C = (learning_rate - 1) / max_iter
        return learning_rate / (1 + (t * C))

    sigma_ = int(min(mn))

    som_gp = MiniSom(
        som_params["m"],
        som_params["n"],
        p_,
        sigma=sigma_,
        learning_rate=som_params["learning_rate"],
        activation_distance="euclidean",
        topology="hexagonal",
        neighborhood_function="gaussian",
        decay_function=inverse_decay_to_zero,
        random_seed=som_params["random_state"],
    )

    gp_vals = df[features_scaled].values
    som_gp.train(gp_vals, som_params["epochs"], use_epochs=True, verbose=True)
    qe = som_gp.quantization_error(gp_vals)
    te = som_gp.topographic_error(gp_vals)

    print(qe, te)

    # save_trained_som(som_gp)

    df["BMU"] = df[features_scaled].apply(
        lambda row: get_row_bmu(row, som_gp, mn), axis=1
    )

    return df, som_gp


def get_row_bmu(row, som, mn):
    """
    Returns the longform ID of the BMU for this row
    """
    row_ = row.values
    return np.ravel_multi_index(som.winner(row.values), mn)


def get_som_cds(df, features_orig):

    features = ["{}{}".format(i, "_std") for i in features_orig]

    som_df, som_gp = get_som(df, features_orig)

    m = som_params["m"]
    n = som_params["n"]

    xx, yy = som_gp.get_euclidean_coordinates()

    weights = som_gp.get_weights()

    umatrix = som_gp.distance_map(scaling="mean")
    umatrix_vals = umatrix.reshape(m * n)
    c = umatrix.reshape(m * n)

    hitsmatrix = som_gp.activation_response(df[features].values)

    hits_flat = hitsmatrix.reshape(m * n)
    umat_flat = umatrix.reshape(m * n)

    hits_pct = hitsmatrix.reshape(m * n) / hitsmatrix.max()

    palette = "Viridis256"

    size = np.min([PLOT_WIDTH / m, PLOT_HEIGHT / n])

    c2 = hitsmatrix.reshape(m * n)
    x2 = xx.reshape(m * n) * size  # / (np.sqrt(3)/2)
    y2 = yy.reshape(m * n) * size

    hits_labels = [i.astype(int).astype(str) if i > 0 else "" for i in c2]
    hits_labels_color = [
        1 if i.astype(int) > (0.75 * np.mean([max(c2), min(c2)])) else 0 for i in c2
    ]

    node_indices = list(range(m * n))
    coords = ["{}".format(np.unravel_index(i, weights.shape[:2])) for i in range(m * n)]

    radius_multiplier = 0.3
    som_cp = {}
    for i in range(len(features)):
        som_cp[features_orig[i]] = som_gp.get_weights()[:, :, i]
    som_cds = ColumnDataSource(
        dict(
            x=x2,
            y=y2,
            c=c2,
            text=hits_labels,
            label_colors=hits_labels_color,
            umatrix=umat_flat,
            _current_hex=umat_flat,
            hits_flat=hits_flat,
            hits_pct=hits_pct * size * 0.75,
            coords=coords,
            node_labels=coords,
            radius=hits_flat * radius_multiplier,
            index=node_indices,
            **som_cp
        )
    )

    return som_df, som_gp, som_cds


def save_trained_som(som, where=None):
    if where == None:
        pickle_path = os.path.join(
            ROOT_PATH, os.path.relpath(os.path.join(".", "pre", "models", "som.p"))
        )
    else:
        pickle_path = where

    with open(pickle_path, "wb") as outfile:
        pickle.dump(som, outfile)
        print("Pickling SOM to", pickle_path)

    return


def get_pretrained_som(where=None):
    if where == None:
        pickle_path = os.path.join(
            ROOT_PATH, os.path.relpath(os.path.join(".", "pre", "models", "som.p"))
        )

    with open(pickle_path, "rb") as infile:
        print("loading pretrained SOM from: ", pickle_path)
        som = pickle.load(infile)

    return som
