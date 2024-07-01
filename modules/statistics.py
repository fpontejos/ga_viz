import datetime
import json
import logging
import os
import re
import sqlite3
import sys
import time
import warnings
from itertools import combinations, permutations

import geopandas as gp
import libpysal as ps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.models import GeoJSONDataSource
from esda.moran import Moran_BV, Moran_Local_BV
from geodatasets import get_path
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import shift_colormap, truncate_colormap
from pysal.explore import esda
from pysal.lib import weights as pysal_weights
from scipy import stats
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from splot.esda import moran_scatterplot
from tqdm.notebook import tqdm, trange

warnings.filterwarnings("ignore")

from constants import *

features = DEFAULT_FEATURES
feats = DEFAULT_FEATS
geo_feats = DEFAULT_GEO_FEATS
stats_csv_path = os.path.join(
    ROOT_PATH, os.path.join("pre", "data", "stats_georgia.csv")
)

lags_csv_path = os.path.join(ROOT_PATH, os.path.join("pre", "data", "lags_georgia.csv"))
details_path = os.path.join(ROOT_PATH, os.path.join("pre", "data", "details.json"))


spots_mapper = {
    0: "NS",
    1: "HH",
    2: "LH",
    3: "LL",
    4: "HL",
}


hilo_cmap = {
    "LH": "#a6cee3",
    "LL": "#1f78b4",
    "HH": "#d93b43",
    "HL": "#e08d49",
    "NS": "#eaeaea",
}


def get_gwr_mgwr(df, geox, geoy, x_feats, y_feats, bw_params):

    g_y = df[y_feats].values.reshape((-1, 1))
    g_X = df[x_feats].values
    u = df[geox]
    v = df[geoy]

    g_coords = list(zip(u, v))

    # Calibrate GWR model
    g_selector = Sel_BW(g_coords, g_y, g_X)
    g_bw = g_selector.search(bw_min=bw_params["min"])
    g_results = GWR(g_coords, g_y, g_X, g_bw).fit()

    # Calibrate MGWR model
    m_selector = Sel_BW(g_coords, g_y, g_X, multi=True)
    m_bw = m_selector.search(multi_bw_min=bw_params["multi"])
    m_results = MGWR(g_coords, g_y, g_X, m_selector).fit()

    return g_results, m_results


def get_gwr_params(feats, results, prefix="gwr", suffix="filter_t"):
    results_feats = [prefix + "_intercept"] + [(prefix + "_" + f) for f in feats]

    res_df = pd.DataFrame(results.params, columns=results_feats)

    ## Mask non-significant t-vals
    filtered_results = results.filter_tvals()
    results_feats_filt = [prefix + "_intercept_" + suffix] + [
        (prefix + "_" + f + "_" + suffix) for f in feats
    ]
    res_df[results_feats_filt] = res_df[results_feats]

    for fi in range(len(results_feats_filt)):
        res_df.loc[filtered_results[:, fi] == 0, results_feats_filt[fi]] = 0

    return res_df


###################################
###################################
## Global Moran's I
###################################
###################################


def get_w(
    df,
    weights_mx_name="KNN",
    weights_params=dict(k=8),
):
    try:
        return WEIGHTS_MX[weights_mx_name].from_dataframe(df, **weights_params)
    except:
        return pysal_weights.KNN.from_dataframe(df, **weights_params)


def get_lags(w, df, feats):

    lags_df = pd.DataFrame()

    for f in feats:
        lags_df[f + "_std"] = df[f + "_std"]
        lags_df[f + "_lag"] = pysal_weights.lag_spatial(w, df[f + "_std"])
        lags_df[f + "_lag_std"] = lags_df[f + "_lag"] - lags_df[f + "_lag"].mean()

    return lags_df


###################################
###################################
## Local Moran's I
###################################
###################################


def get_lisa(w, df, lags_df, feats, p=0.05):

    lisa_df = pd.DataFrame()
    lisa_models = {}

    for f in feats:
        lisa_f = f + "_lisa"
        lisa_fc = lisa_f + "_clust"
        lisa_fcf = lisa_fc + "_filt"

        lisa = esda.moran.Moran_Local(df[f], w)
        lisa_df[lisa_f] = lisa.Is
        lisa_df[lisa_fc] = lisa.q

        lisa_models[f] = lisa

        sig = lisa.p_sim >= p
        lisa_df[lisa_fcf] = lisa.q
        lisa_df.loc[sig, lisa_fcf] = 0
        lisa_df[lisa_fcf] = lisa_df[lisa_fcf].map(spots_mapper)

        # Take the standardized values
        lisa_df[f + "_std"] = df[f + "_std"]

        # Take the lag of std
        lisa_df[f + "_lag"] = pysal_weights.lag_spatial(w, lisa_df[f + "_std"])

        # Standardize the lag
        lisa_df[f + "_lag_std"] = lisa_df[f + "_lag"] - lags_df[f + "_lag"].mean()

    return lisa_df, lisa_models


###################################
###################################
## Moran Bivariate Statistics
###################################
###################################


def get_moran_bv(df, feats, w, spots_mapper=spots_mapper, p=0.05):

    f_combis = list(permutations(feats, 2))

    mbv_df = pd.DataFrame()

    mbv_models = {}

    for fi in f_combis:
        mbv_name = "MBV_" + "_".join(fi)

        mbv = Moran_Local_BV(df["{}_std".format(fi[0])], df["{}_std".format(fi[1])], w)
        mbv_models[mbv_name] = mbv

        mbv_df[mbv_name] = mbv.q
        sig = mbv.p_sim >= p
        mbv_df.loc[sig, mbv_name] = 0
        mbv_df[mbv_name] = mbv_df[mbv_name].map(spots_mapper)

    return mbv_df, mbv_models


def get_statistics_df(
    gp_df,
    features=DEFAULT_FEATURES,
    feats=DEFAULT_FEATS,
    geo_feats=DEFAULT_GEO_FEATS,
    use_precalc=False,
):
    orig_feats = gp_df.columns.to_list()

    if use_precalc == True:
        stats_df = pd.read_csv(stats_csv_path)
        gp_df = pd.concat([gp_df, stats_df], axis=1)
        lags_df = pd.read_csv(lags_csv_path)

        with open(details_path) as f:
            details = json.load(f)

    else:

        # Scale features using SS
        features_scaled = ["{}{}".format(i, feats["suf"]["scaled"]) for i in features]

        # Get features subset
        feats["x"] = ["PctFB", "PctBlack", "PctRural"]
        feats["y"] = "PctBach"

        feats_yx = [feats["y"]] + feats["x"]

        # End features subset

        # Populate jitter columns for boxplots later
        gp_df["_jitter"] = 1

        for f in feats_yx:
            gp_df["{}_jit".format(f)] = f
        # End jitter
        details = {}

        gp_df_gwr = gp_df.copy()

        stats_params = {}

        stats_params["bw"] = {"min": 2, "multi": [2]}

        ## Get scaled features

        scaled_x_feats = ["{}{}".format(i, feats["suf"]["scaled"]) for i in feats["x"]]
        scaled_y_feat = "{}{}".format(feats["y"], feats["suf"]["scaled"])

        ###################################
        ## Fit GWR/MGWR models
        ###################################

        gwr_results, mgwr_results = get_gwr_mgwr(
            gp_df,
            feats["geo"]["x"],
            feats["geo"]["y"],
            scaled_x_feats,
            scaled_y_feat,
            stats_params["bw"],
        )

        ###################################
        ## Get GWR Coefficients
        ## Get MGWR Coefficients
        ###################################

        gwr_mgwr_df = pd.concat(
            [
                get_gwr_params(feats["x"], gwr_results),
                get_gwr_params(feats["x"], mgwr_results, prefix="mgwr"),
            ],
            axis=1,
        )

        ###################################
        ## Get stat summaries
        ###################################

        details["gwr_summary"] = gwr_results.summary(as_str=True)
        details["mgwr_summary"] = mgwr_results.summary(as_str=True)

        gm_cols = [i for i in gwr_mgwr_df.columns if i not in gp_df.columns]

        gp_df = pd.concat([gp_df, gwr_mgwr_df[gm_cols]], axis=1)

        ###################################
        ## Global Moran
        ###################################

        gp_df_moran = gp_df.copy()

        w = get_w(gp_df)

        # Row-standardization
        w.transform = "R"

        ## Get spatial lags
        lags_df = get_lags(w, gp_df, [feats["y"]] + feats["x"])
        lags_reg = linregress(gp_df["PctBlack_std"], lags_df["PctBlack_lag_std"])

        features_scaled = ["{}{}".format(i, feats["suf"]["scaled"]) for i in features]

        lags_reg = linregress(lags_df["PctBlack_std"], lags_df["PctBlack_lag_std"])
        keep_cols_lags = [c for c in lags_df.columns if c not in gp_df_moran.columns]

        gp_df = pd.concat(
            [gp_df_moran, lags_df[keep_cols_lags]], axis=1
        ).drop_duplicates()

        ###################################
        ## Local Moran
        ###################################

        gp_df_lisa = gp_df.copy()

        lisa_df, lisa_models = get_lisa(
            w, gp_df_lisa, lags_df, [feats["y"]] + feats["x"]
        )

        keep_cols_lisa = [c for c in lisa_df.columns if c not in gp_df_lisa.columns]

        gp_df = pd.concat(
            [gp_df_lisa, lisa_df[keep_cols_lisa]], axis=1
        ).drop_duplicates()

        ###################################
        ## Moran Bivariate
        ###################################

        gp_df_mbv = gp_df.copy()

        mbv_df, mbv_models = get_moran_bv(gp_df, [feats["y"]] + feats["x"], w)

        gp_df = pd.concat([gp_df_mbv, mbv_df], axis=1)

        new_feats = [i for i in gp_df.columns if i not in orig_feats]
        gp_df[new_feats].to_csv(stats_csv_path, index=False)
        lags_df.to_csv(lags_csv_path, index=False)

        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=4)

    gp_df["_current_geo"] = gp_df[feats["y"] + "_std"]
    gp_df["_current_x"] = gp_df[feats["y"] + "_std"]
    gp_df["_current_y"] = gp_df[feats["y"] + "_lag_std"]
    gp_df["_ns"] = "NS"
    gp_df["_current_c"] = gp_df["_ns"]

    geosource = GeoJSONDataSource(geojson=gp_df.to_json())

    return gp_df, feats, geosource, lags_df, details
