import argparse

from bokeh.io import curdoc
from bokeh.models import Select
from constants import *
from modules.callbacks.js_callbacks import *
from modules.clustering import *
from modules.setup import *
from modules.statistics import *
from modules.visualization import *

parser = argparse.ArgumentParser()
parser = setup_parser(parser)
args = parser.parse_args()


# curdoc().add_root(details_tab_layout)
# curdoc().add_root(plot)
# curdoc().add_root(colorbar_plot)

# curdoc().add_root(footer_overlay)
# curdoc().add_root(footer_input)
# curdoc().add_root(footer_user)

features = DEFAULT_FEATURES
feats = DEFAULT_FEATS
geo_feats = DEFAULT_GEO_FEATS


def get_data(
    features=DEFAULT_FEATURES, feats=DEFAULT_FEATS, geo_feats=DEFAULT_GEO_FEATS
):

    gp_georgia = gp.read_file(ps.examples.get_path("G_utm.shp"))
    gp_df = gp_georgia.copy()

    # Scale features using SS
    features_scaled = ["{}{}".format(i, feats["suf"]["scaled"]) for i in features]

    st_scaler = StandardScaler()
    gp_df[features_scaled] = st_scaler.fit_transform(gp_df[features])
    # End scaling

    return gp_df


gp_df = get_data()

gp_df, som_gp, som_cds = get_som_cds(gp_df, features)

gp_df, feats, geo_cds, lags_df, details = get_statistics_df(gp_df, use_precalc=True)

features_scaled = ["{}{}".format(i, feats["suf"]["scaled"]) for i in features]
feats_yx = [feats["y"]] + feats["x"]

(
    som_plot,
    som_cbar,
    geo_plot,
    scatter_plot,
    box_plot,
    to_dropdown,
    radio_opts,
    som_buttons,
    geo_btns,
    text_plot,
) = make_viz(som_cds, geo_cds, gp_df, lags_df, feats_yx, details)


curdoc().add_root(box_plot)
curdoc().add_root(som_plot)
curdoc().add_root(som_cbar)
curdoc().add_root(geo_plot)
curdoc().add_root(scatter_plot)
curdoc().add_root(radio_opts)
curdoc().add_root(som_buttons)
curdoc().add_root(geo_btns)
# curdoc().add_root(text_plot)


# curdoc().add_root(to_dropdown)

curdoc().title = "Interactive Visual Analytics for Spatial Data"
