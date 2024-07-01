import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    AllIndices,
    BasicTicker,
    CDSView,
    CheckboxButtonGroup,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    FactorRange,
    HoverTool,
    IndexFilter,
    LinearColorMapper,
    RadioButtonGroup,
    Range1d,
    Select,
    Slope,
    Whisker,
)
from bokeh.palettes import Viridis6 as palette
from bokeh.palettes import interp_palette
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, jitter
from constants import *
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mpl_colors
from modules.callbacks.js_callbacks import *
from scipy.stats import linregress

som_params = DEFAULT_SOM_PARAMS


def make_viz(som_cds, geo_cds, gp_df, lags_df, feats_yx, details):

    umatrix_cmap = LinearColorMapper(
        palette=palette, low=0, high=max(som_cds.data["umatrix"])
    )

    seismic = [mpl_colormaps["seismic"].resampled(256)(i) for i in range(256)]
    seismic = [mpl_colors.to_hex(i) for i in seismic]
    geo_palette = interp_palette(seismic, 256)
    geo_cmap_col = "_current_geo"

    geo_all = AllIndices()
    geo_view = CDSView(filter=geo_all)

    som_all = AllIndices()
    som_view = CDSView(filter=som_all)

    som_plot, som_colorbar, som_cmap = plot_hexagons(som_cds, som_params)
    geo_plot, geo_color_bar, geo_shapes, geo_cmap = plot_geo(
        gp_df, geo_cds, geo_cmap_col, geo_palette
    )
    scatter_plot, to_dropdown, radio_opts, som_buttons, geo_btns = plot_scatter(
        geo_cds,
        lags_df,
        feats_yx,
        geo_shapes,
        geo_cmap,
        geo_plot,
        som_cds,
        som_plot,
        som_cmap,
    )

    box_plot = plot_boxplots(gp_df, feats_yx, geo_cds)

    som_cds, geo_cds = plot_callbacks(
        som_cds, geo_cds, som_plot, geo_plot, som_view, geo_view, scatter_plot
    )

    return (
        som_plot,
        som_colorbar,
        geo_plot,
        scatter_plot,
        box_plot,
        to_dropdown,
        radio_opts,
        som_buttons,
        geo_btns,
        details,
    )


def plot_hexagons(som_cds, som_params):

    m = som_params["m"]
    n = som_params["n"]
    palette = "Viridis256"

    umatrix_cmap = LinearColorMapper(
        palette=palette,
        low=min(0, min(som_cds.data["_current_hex"])),
        high=max(1, max(som_cds.data["_current_hex"])),
    )

    size = np.min([PLOT_WIDTH / m, PLOT_HEIGHT / n])

    plot = figure(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        tools="tap,reset,save",
        toolbar_location="above",
        match_aspect=True,
        aspect_scale=1,
        title="SOM Grid Map",
        outline_line_width=0,
        name="som_plot",
    )
    plot.toolbar.logo = None

    bg_all = AllIndices()
    bg_view = CDSView(filter=bg_all)

    som_cmap = umatrix_cmap
    bg_hex = plot.scatter(
        source=som_cds,
        size=size * 1.05,
        marker="hex",
        line_join="miter",
        fill_color={"field": "_current_hex", "transform": som_cmap},
        line_color={"field": "_current_hex", "transform": som_cmap},
        line_width=1,
        angle=90,
        angle_units="deg",
        alpha=1,
        hover_fill_alpha=1,
        hover_line_width=4,
        hover_line_color=CONTRAST_COLOR1,
        nonselection_alpha=0.25,
        selection_line_color=CONTRAST_COLOR1,
        selection_fill_alpha=1.0,
        selection_line_width=6,
        view=bg_view,
        name="hex_shape",
    )

    um_colorbar = ColorBar(
        color_mapper=som_cmap,
        location=(0, 0),
        width=10,
        height=int(PLOT_HEIGHT / 2),
        ticker=BasicTicker(min_interval=0.1),
    )

    hits_hex = plot.scatter(
        source=som_cds,
        size="hits_pct",
        marker="hex",
        line_join="miter",
        fill_color="#fafafa",
        line_width=0,
        angle=90,
        angle_units="deg",
        alpha=1,
        nonselection_alpha=1,
        selection_alpha=1,
    )

    tooltips = [("HITS", "@umatrix"), ("Average Dist", "@umatrix")]

    plot.add_tools(HoverTool(tooltips=tooltips, renderers=[bg_hex]))
    plot.grid.visible = False
    plot.axis.visible = False

    colorbar_plot = figure(
        width=60,  # height = PLOT_HEIGHT,
        toolbar_location=None,
        outline_line_width=0,
        name="colorbar_plot",
    )

    colorbar_plot.add_layout(um_colorbar, "right")

    return plot, colorbar_plot, som_cmap


def plot_geo(gp_df, geo_cds, geo_cmap_col, geo_palette):

    lx_cmap = LinearColorMapper(
        palette=geo_palette,
        # low=-1,
        # high=1,
        low=np.min([0, gp_df[geo_cmap_col].min()]),
        high=np.max([0, gp_df[geo_cmap_col].max()]),
    )

    geo_plot = figure(
        title="Georgia Dataset",
        aspect_ratio=1,
        match_aspect=True,
        height=PLOT_HEIGHT,
        tools="tap,reset,save",
        name="geo_plot",
    )

    geo_plot.xgrid.grid_line_color = None
    geo_plot.ygrid.grid_line_color = None

    geo_all = AllIndices()
    geo_view = CDSView(filter=geo_all)

    pt_shapes = geo_plot.patches(
        "xs",
        "ys",
        source=geo_cds,
        fill_color={"field": geo_cmap_col, "transform": lx_cmap},
        line_color="black",
        selection_line_color="orange",
        line_width=1,
        selection_line_width=4,
        fill_alpha=0.75,
        nonselection_fill_alpha=0.5,
        view=geo_view,
        name="geo_units",
    )

    geo_plot.axis.visible = False

    geo_plot.add_tools(HoverTool(renderers=[pt_shapes]))

    geo_color_bar = pt_shapes.construct_color_bar(
        major_tick_line_color=None,
    )

    geo_plot.add_layout(geo_color_bar, "right")

    return geo_plot, geo_color_bar, pt_shapes, lx_cmap


def plot_callbacks(
    som_cds, geo_cds, som_plot, geo_plot, som_view, geo_view, scatter_plot
):

    som_cds.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                geo_source=geo_cds,
                som_cds=som_cds,
                geo_plot=geo_plot,
                geo_view=geo_view,
                geo_all=AllIndices(),
                geo_index=IndexFilter(indices=[]),
            ),
            code=CALLBACKS["som_cb"],
        ),
    )

    # TODO:
    # Callback for clicking on a geo unit that will add an indicator to its BMU

    geo_cds.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                geo_source=geo_cds,
                som_cds=som_cds,
                geo_plot=geo_plot,
                geo_view=geo_view,
                geo_all=AllIndices(),
                geo_index=IndexFilter(indices=[]),
                splot=scatter_plot,
                hex_view=som_view,
                hex_all=AllIndices(),
                hex_index=IndexFilter(indices=[]),
            ),
            code=CALLBACKS["geo_cb"],
        ),
    )

    return som_cds, geo_cds


def plot_dd(
    features,
    target_fig,
    target_glyph,
    geo_cds,
    geo_shapes,
    geo_cmap,
    geo_plot,
    som_cds,
    som_plot,
    som_cmap,
):

    to_dropdown = Select(
        value=features[0],
        options=features,
        visible=True,
        min_width=200,
        sizing_mode="stretch_width",
        styles=dict({"min-width": "120px"}),
        name="to_dropdown",
    )

    LABELS = [
        "GWR",
        "MGWR",
        "Global Moran",
        "Local Moran",
        "Bivariate Local Moran",
        "Features",
    ]

    geo_feats = [i for i in features] + ["GWR", "MGWR"]
    geo_labels = ["Original", "Scaled", "Sig. GWR", "Sig. MGWR"]

    geo_btns1 = RadioButtonGroup(labels=geo_feats, active=0, name="geo_btns1")
    geo_btns2 = RadioButtonGroup(labels=geo_labels, active=0, name="geo_btns2")

    geo_btns_cb = CustomJS(
        args=dict(
            btn_group="stats",
            geo_cds=geo_cds,
            splot=target_fig,
            rdots=target_glyph,
            selectx=to_dropdown,
            selecty=to_dropdown,
            geo_btns=geo_btns1,
            rad1=geo_btns1,
            rad2=geo_btns2,
            rad3=geo_btns2,
            labels=geo_labels,
            features=geo_feats,
            splotx=target_fig.xaxis,
            sploty=target_fig.yaxis,
            geo_shapes=geo_shapes,
            geo_cmap=geo_cmap,
            geo_plot=geo_plot,
        ),
        code="""
        
        console.log(labels)
        console.log(features)

        let geo_feat = features[rad1.active]
        let geo_filt = labels[rad2.active]

        let current_feat = geo_feat

        if (rad1.active ==4) {
            current_feat = 'gwr_intercept'
        } else if (rad1.active ==5) {
            current_feat = 'mgwr_intercept'
        } 
        console.log(1, current_feat)

        

        if (rad2.active == 1) {
            current_feat = geo_feat + '_std'
        } else if (rad2.active == 2) {
            current_feat = 'gwr_' + current_feat + '_filter_t'
        } else if (rad2.active == 3) {
            current_feat = 'mgwr_' + current_feat + '_filter_t'
        } 
        console.log(2, current_feat)
       
        geo_cds.data['_current_geo'] = geo_cds.data[current_feat]

        geo_plot.title.text = geo_feat
        
        var minz = Math.floor(Math.min(...geo_cds.data['_current_geo']))
        var maxz = Math.ceil(Math.max(...geo_cds.data['_current_geo']))

        var absmin = Math.min(Math.abs(minz), Math.abs(maxz))
        var absmax = Math.max(Math.abs(minz), Math.abs(maxz))

        geo_cmap.low=(-1*absmax)
        geo_cmap.high=absmax

        geo_cds.change.emit();

                        """,
    )

    geo_btns1.js_on_event("button_click", geo_btns_cb)
    geo_btns2.js_on_event("button_click", geo_btns_cb)

    radio_button_group1 = RadioButtonGroup(labels=LABELS[2:], active=0, name="rbg1")

    radio_button_group2 = RadioButtonGroup(labels=features, active=0)
    radio_button_group3 = RadioButtonGroup(labels=features, active=0)

    btn_cb = CustomJS(
        args=dict(
            btn_group="stats",
            geo_cds=geo_cds,
            splot=target_fig,
            rdots=target_glyph,
            selectx=to_dropdown,
            selecty=to_dropdown,
            rad1=radio_button_group1,
            rad2=radio_button_group2,
            rad3=radio_button_group3,
            labels=LABELS[2:],
            features=features,
            splotx=target_fig.xaxis,
            sploty=target_fig.yaxis,
        ),
        code="""

        console.log(cb_obj.origin.name)

        var stat_type = rad1.active
        var xvar = features[rad2.active]
        var yvar = features[rad3.active]

        var x_field=""
        var y_field=""
        var c_field = "_ns"
        splotx[0].axis_label = xvar
        sploty[0].axis_label = yvar

        //splot.title.text = labels[rad1.active]": " + xvar 
        
        if ((rad1.active==0) || (rad1.active==1)) {
            x_field = xvar + '_std'
            y_field = xvar + '_lag_std'
            if (rad1.active==1) {
                c_field = xvar + '_lisa_clust_filt'
            }

            splotx[0].axis_label = xvar
            sploty[0].axis_label = xvar + " Lag"
            console.log(rad2, 'rad2', xvar, yvar)
            rad3.visible = false
            rad3.change.emit()
        } else if (rad1.active==2) {
            x_field = xvar + '_std'
            y_field = yvar + '_lag_std'
            c_field = 'MBV_' + xvar + '_' + yvar

            rad3.visible = true
            rad3.change.emit()
            
        } else {
            x_field = xvar + '_std'
            y_field = yvar + '_std'

        }


        console.log(geo_cds)
        /*
        const yf = selecty.value + '_lag_std'
        const xf = selectx.value + '_std'
        const cf = selectx.value + '_lisa_clust_filt'
        console.log(selectx.value, selecty.value)
        */

        geo_cds.data['_current_x'] = geo_cds.data[x_field]
        geo_cds.data['_current_y'] = geo_cds.data[y_field]
        geo_cds.data['_current_c'] = geo_cds.data[c_field]

        rdots.glyph.fill_color['field'] = c_field
        rdots.glyph.line_color['field'] = c_field
        


        geo_cds.change.emit();

                        """,
    )

    radio_button_group1.js_on_event("button_click", btn_cb)
    radio_button_group2.js_on_event("button_click", btn_cb)
    radio_button_group3.js_on_event("button_click", btn_cb)

    dd_cb = CustomJS(
        args=dict(
            geo_cds=geo_cds,
            splot=target_fig,
            rdots=target_glyph,
            selectx=to_dropdown,
            geo_shapes=geo_shapes,
            geo_cmap=geo_cmap,
            geo_plot=geo_plot,
        ),
        code="""

    console.log(geo_cds)
    console.log(geo_cmap)
    
    const xf = selectx.value + '_std'
    const cf = selectx.value + '_lisa_clust_filt'
    
    geo_cds.data['_current_geo'] = geo_cds.data[xf]

    geo_plot.title.text = selectx.value
    
    var minz = Math.floor(Math.min(...geo_cds.data['_current_geo']))
    var maxz = Math.ceil(Math.max(...geo_cds.data['_current_geo']))


    var absmin = Math.min(Math.abs(minz), Math.abs(maxz))
    var absmax = Math.max(Math.abs(minz), Math.abs(maxz))

    geo_cmap.low=(-1*absmax)
    geo_cmap.high=absmax

    
    console.log(rdots.glyph)

    geo_cds.change.emit();

                    """,
    )

    # to_dropdown.on_change("value", cb_handler)
    to_dropdown.js_on_change("value", dd_cb)

    radio_opts = column(
        radio_button_group1,
        radio_button_group2,
        radio_button_group3,
        name="radio_opts",
    )

    som_feats = ["UMatrix"] + features
    som_buttons = RadioButtonGroup(labels=som_feats, active=0, name="som_btns")

    som_glyph = som_plot.select_one("hex_shape")  # .glyph.x = "PctBach"
    som_cb = CustomJS(
        args=dict(
            som_cds=som_cds,
            splot=som_plot,
            som_glyph=som_glyph,
            geo_shapes=geo_shapes,
            som_cmap=som_cmap,
            som_feats=som_feats,
            geo_plot=geo_plot,
            som_buttons=som_buttons,
        ),
        code="""

        console.log(som_cds)
        console.log(som_cmap)
        console.log(som_glyph)
        
        var xf = som_feats[som_buttons.active]
        if (som_buttons.active == 0) {
            xf = "umatrix"
        }
        som_cds.data['_current_hex'] = som_cds.data[xf]

        splot.title.text = xf
        
        var minz = Math.floor(Math.min(...som_cds.data['_current_hex']))
        var maxz = Math.ceil(Math.max(...som_cds.data['_current_hex']))
        som_cmap.low=minz
        som_cmap.high=maxz
        
        console.log(som_glyph.glyph)

        som_cds.change.emit();

                        """,
    )
    som_buttons.js_on_event("button_click", som_cb)

    geo_btns = column(geo_btns1, geo_btns2, name="geo_btns")

    return to_dropdown, radio_opts, som_buttons, geo_btns


def plot_scatter(
    geo_cds,
    lags_df,
    features,
    geo_shapes,
    geo_cmap,
    geo_plot,
    som_cds,
    som_plot,
    som_cmap,
):

    scatter_feat = features[0]
    scatter_feat_x = "{}_std".format(scatter_feat)
    scatter_feat_y = "{}_lag_std".format(scatter_feat)

    curr_feat_x = "_current_x"
    curr_feat_y = "_current_y"
    curr_feat_c = "_current_c"

    hilo_cmap = {
        "NS": "#a9a9a9",
        "LL": "#1f78b4",
        "LH": "#7696A7",
        "HL": "#e08d49",
        "HH": "#d93b43",
    }

    scatter_plot = figure(
        title="Scatter Plot",
        x_axis_label=scatter_feat_x,
        y_axis_label=scatter_feat_y,
        x_range=DataRange1d(),
        y_range=DataRange1d(),
        aspect_ratio=1,
        match_aspect=True,
        height=int(PLOT_HEIGHT * 0.7),
        tools="tap,reset,save,box_select",
        name="scatter_plot",
    )

    lags_reg = linregress(lags_df[scatter_feat_x], lags_df[scatter_feat_y])

    scatter_plot.hspan(y=[0], line_color="black")
    scatter_plot.vspan(x=[0], line_color="black")

    slope = Slope(
        gradient=lags_reg.slope,
        y_intercept=lags_reg.intercept,
        line_color="red",
        line_dash="dashed",
        line_width=4,
    )

    scatter_dots = scatter_plot.circle(
        x=curr_feat_x,
        y=curr_feat_y,
        source=geo_cds,
        radius=0.05,
        # selection_line_color="#2F2F2F",
        # selection_line_width=3,
        line_width=1,
        fill_alpha=0.7,
        line_alpha=1,
        nonselection_fill_alpha=0.25,
        nonselection_line_alpha=0.5,
        selection_fill_alpha=1,
        selection_line_color="#5e4c54",
        fill_color=factor_cmap(
            curr_feat_c,
            palette=list(hilo_cmap.values()),
            factors=list(hilo_cmap.keys()),
        ),
        line_color=factor_cmap(
            curr_feat_c,
            palette=list(hilo_cmap.values()),
            factors=list(hilo_cmap.keys()),
        ),
        name="scatter_dots",
    )

    # scatter_plot.add_layout(slope)

    scatter_plot.add_tools(HoverTool(renderers=[scatter_dots]))

    scatter_color_bar = scatter_dots.construct_color_bar(
        major_tick_line_color=None,
    )

    scatter_plot.add_layout(scatter_color_bar, "right")

    to_dropdown, radio_opts, som_buttons, geo_btns = plot_dd(
        features,
        scatter_plot,
        scatter_dots,
        geo_cds,
        geo_shapes,
        geo_cmap,
        geo_plot,
        som_cds,
        som_plot,
        som_cmap,
    )

    return scatter_plot, to_dropdown, radio_opts, som_buttons, geo_btns


def plot_boxplots(df, feats, box_cds):
    boxplot_figs = []
    jit_feats = ["{}_jit" for i in feats]

    iqr_df = (
        df[feats].describe().T.rename(columns={"25%": "q1", "50%": "q2", "75%": "q3"})
    )

    iqr_df["iqr"] = (iqr_df["q3"] - iqr_df["q1"]) * 1.5
    iqr_df["upper"] = iqr_df["q3"] + iqr_df["iqr"]
    iqr_df["lower"] = iqr_df["q1"] - iqr_df["iqr"]
    iqr_df["upper"] = iqr_df.apply(
        lambda row: np.min([row["max"], row["upper"]]), axis=1
    )
    iqr_df["lower"] = iqr_df.apply(
        lambda row: np.max([row["min"], row["lower"]]), axis=1
    )

    iqr_df.reset_index(inplace=True)

    iqr_cds = ColumnDataSource(iqr_df)

    for f in feats:

        univar_fig = figure(
            height=400,
            width=100,
            x_range=[f],
            title=f,
            toolbar_location=None,
            tools="tap,reset,box_select",
        )
        univar_fig.xgrid.grid_line_color = None
        univar_fig.ygrid.grid_line_color = None

        univar_fig.x_range = FactorRange(factors=[f])

        error = Whisker(
            base="index",
            upper="upper",
            lower="lower",
            source=iqr_cds,
            line_width=2,
            level="annotation",
        )
        error.upper_head.size = 20
        error.lower_head.size = 20
        univar_fig.add_layout(error)

        univar_fig.vbar(
            "index",
            0.5,
            "q2",
            "q3",
            source=iqr_cds,
            color="#AAAAAA",
            line_color="black",
            level="overlay",
        )
        univar_fig.vbar(
            "index",
            0.5,
            "q1",
            "q2",
            source=iqr_cds,
            color="#AAAAAA",
            level="overlay",
            line_color="black",
        )

        univar_fig.scatter(
            x=jitter("{}_jit".format(f), width=0.45, range=univar_fig.x_range),
            y=f,
            source=box_cds,
            alpha=0.5,
            size=5,
            selection_color="orange",
            selection_line_width=3,
            line_width=1,
            level="overlay",
            color="#1f7899",
            fill_alpha=0.25,
            line_alpha=0.85,
            nonselection_fill_alpha=0.25,
            selection_fill_alpha=0.75,
        )
        boxplot_figs.append(univar_fig)

    boxplots = row(*boxplot_figs, name="box_plot")

    return boxplots
