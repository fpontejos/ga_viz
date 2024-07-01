from bokeh.models import PrintfTickFormatter
from bokeh.models import ColorBar, BasicTicker
from bokeh.plotting import figure

from constants import *

def make_colorbar_plot(plot, cmap, pos='right'):
    
    hex_colorbar = ColorBar(
                        color_mapper=cmap, 
                        location=(0,0),
                        width=10,
                        height=200,
                        ticker=BasicTicker(min_interval=.1)
                        )

    #plot.yaxis.formatter = PrintfTickFormatter(format='%.1f')
    plot.add_layout(hex_colorbar, pos)

    return plot, hex_colorbar