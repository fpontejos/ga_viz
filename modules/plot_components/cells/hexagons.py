import numpy as np

from bokeh.models import ColumnDataSource, CDSView

from bokeh.models import ColorBar, HoverTool, BasicTicker

from bokeh.models import IndexFilter
from bokeh.models import ColumnDataSource, CDSView, CustomJS, Styles

from bokeh.models import TabPanel, Tabs, Switch

from modules.plot_components.js_callbacks import *
from constants import * 

def make_user_hex(plot, user_src, user_view, size):

    plot.scatter(source=user_src,
            size=size,
            marker="hex", 
            line_join='miter',
            fill_color=None,
            line_width=0,
            hatch_pattern='+',
            hatch_color='#ffffff',
            hatch_weight=1,
            hatch_scale=4,#1.5,
            hatch_alpha=.9,
            nonselection_hatch_alpha=.8,
            selection_hatch_alpha=1,
            angle=90, 
            angle_units='deg',
            alpha=1,
            view=user_view
    )



    return plot


def make_bg_hex(plot, main_source, cmap, size):

    bg_hex = plot.scatter(source=main_source,
                    size=size+2,
                    marker="hex", 
                    line_join='miter',
                    fill_color={"field":"umat_flat", "transform":cmap},
                    line_color={"field":"umat_flat", "transform":cmap},
                    #line_color="red",
                    line_width=1,
                    angle=90, 
                    angle_units='deg',
                    alpha=1,
                    #hover_fill_color="#eeeeee",
                    hover_fill_alpha=1,
                    hover_line_width=4,
                    hover_line_color=CONTRAST_COLOR1,
                    nonselection_alpha=.25,
                    selection_line_color=CONTRAST_COLOR1,
                    selection_fill_alpha=1.0,
                    selection_line_width=6,

    )

    um_colorbar = ColorBar(
                            color_mapper=cmap, 
                            location=(0,0),
                            width=10,
                            height=500,
                            ticker=BasicTicker(min_interval=.1)
                            )
    return plot, bg_hex, um_colorbar


def make_query_hex(plot, m, n, x2, y2, size):

    highlight_nodes = np.arange(m*n)
    highlight_rank = np.arange(m*n)

    highlight_rank = np.zeros(m*n)


    highlight_src = ColumnDataSource(dict(
            x=x2,
            y=y2,
            rank=highlight_rank,
            highlight=highlight_nodes
        )
    )
    highlight_view = CDSView(filter=IndexFilter(indices=[]))

    # highlight hex plot
    highlight_hex = plot.scatter(source=highlight_src,
                        size=size,
                        marker="hex", 
                        line_join='miter',
                        line_color="#FFF",
                        selection_line_alpha='rank',
                        nonselection_line_alpha='rank',                        
                        line_width=4,
                        fill_color=None,
                        angle=90, 
                        angle_units='deg',
                        nonselection_alpha='rank',
                        alpha='rank',
                        view=highlight_view
    )



    
    return plot, highlight_src, highlight_hex


def make_hits_hex(plot, main_source):
    hits_hex = plot.scatter(source=main_source,
                size='hits_pct',
                marker="hex", 
                line_join='miter',
                fill_color="#fafafa",
                line_width=0,
                angle=90, 
                angle_units='deg',
                alpha=1,
                nonselection_alpha=1,
                selection_alpha=1,
           
    )

    hits_style = """
    :host(.active) .bar{background-color:#c6e6d8;}
    :host(.active) .knob{background-color:#31b57a;}
    """
    hits_switch = Switch(active=True, name="hits_switch", stylesheets=[hits_style])
    hits_switch.js_on_change("active", 
                            CustomJS(
                                    args=dict(hits=hits_hex),
                                    code="""hits.visible = !hits.visible"""))


    return plot, hits_hex, hits_switch

def make_qe_hex(plot, main_source, cmap, size):

    qe_hex = plot.scatter(source=main_source,
                        size=size+2,
                        marker="hex", 
                        line_join='miter',
                        fill_color={"field":"node_errs", "transform":cmap},
                        line_color={"field":"node_errs", "transform":cmap},
                        line_width=1,
                        angle=90, 
                        angle_units='deg',
                        alpha=1,
                        hover_fill_alpha=1,
                        hover_line_width=4,
                        hover_line_color=CONTRAST_COLOR1,
                        nonselection_alpha=.25,
                        selection_line_color=CONTRAST_COLOR1,
                        selection_fill_alpha=1.0,
                        selection_line_width=6,
                        visible=False

        )
    qe_colorbar = ColorBar(
                            color_mapper=cmap, 
                            location=(0,0),
                            width=10,
                            height=500,
                            ticker=BasicTicker(min_interval=.1),
                            visible=False,
                            )
    return plot, qe_hex, qe_colorbar

def annotate_plot(plot, cmap, plot_renderers):

    plot.grid.visible = False
    plot.axis.visible = False

    tooltips = [
        ("HITS", "@hits_flat"),
        ("Average Dist", "@umat_flat")
    ]

    plot.add_tools(HoverTool(tooltips=tooltips, renderers = plot_renderers))

    return plot