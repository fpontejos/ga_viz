import pandas as pd


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CDSView, CustomJS
from bokeh.models import DataTable, TableColumn

from bokeh.models import Div

from bokeh.models import AllIndices, GroupFilter, IndexFilter


from modules.plot_components.js_callbacks import *
from constants import *

def make_bars(summary_embeddings_df):
    ## lists each bmu in which the user is present
    bar_df_ = summary_embeddings_df.loc[:,['target', 'bmu']].copy().reset_index(drop=True)

    ## creats a matrix of bmu x user 
    ## where each intersection contains how many times the user appears in that bmu
    bar_df = pd.crosstab(bar_df_.bmu, bar_df_.target)
    ## stringify the bmu node index labels
    #bar_df.index = bar_df.index.map(str)

    ## transpose the bmu x user matrix and reset index to get a 'target' column
    bar_df_t = bar_df.T
    bar_df_t.reset_index(inplace=True)


    ## create a list of each combination of user and bmu and their intersections
    bar_df_long = bar_df.melt(ignore_index=False).reset_index()
    bar_df_long = bar_df_long[bar_df_long['value']>0].copy().reset_index(drop=True)
    ### SAME AS:
    #bar_df_['count'] = 1
    #bar_df_.groupby(['target','bmu']).count().reset_index().rename(columns={'count':'value'})


    bar_df_long.bmu = bar_df_long.bmu.astype(str)
    bar_df_long.sort_values(by='value', ascending=False, inplace=True)

    ## bar_df_long is a df containing each combination of bmu x user 
    ## and how many times this intersection occurs
    # bar_df_long

    ## Placeholders for index / view values and CDS filters
    index_0 = bar_df_long.loc[bar_df_long['bmu']=='0',:].index.to_list()
    bar_df_long_src = ColumnDataSource(bar_df_long)
    long_index = IndexFilter(indices=index_0)
    long_view = CDSView(filter=long_index)



    bar_df_idx = bar_df_long.loc[bar_df_long['bmu']=='0'].index.tolist()

    ## Get top 10 users by BR count to display bar chart when no node is selected
    top10_users_bar = bar_df_long.groupby('target').sum(numeric_only=True).sort_values('value', ascending=False)[:10]
    top10_users_bar.reset_index(inplace=True)
    top10_users_bar['bmu'] = 'top'

    bar_df_long = pd.concat([bar_df_long,top10_users_bar], ignore_index=True)



    top10index = bar_df_long.loc[bar_df_long['bmu']=='top',:].index.to_list()

    bar_source = ColumnDataSource(data=bar_df_long)
    bar_index = IndexFilter(indices=top10index)
    bar_view = CDSView(filter=bar_index)

    x_range = bar_df_long.iloc[top10index]['target'].tolist()

    ## Initialize bar plot
    bar_plot = figure(
        height=200,
        x_range=x_range,
        tooltips='@target: @value',
        title="Top Users"
        )

    bar_plot.toolbar.logo = None
    bar_plot.toolbar_location = None

    bars = bar_plot.vbar(x='target', 
                source=bar_source, 
                width=.9, 
                view=bar_view,
                top='value',
                hover_fill_color="#bebebe", 
                fill_color="#959595",
                line_width= 0,
                )
    bar_plot.grid.visible = False
    bar_plot.axis.minor_tick_line_color = None
    bar_plot.xaxis.visible = False


    bar_plot.css_classes=["bar_plot"]
    bar_plot.sizing_mode="stretch_width"
    bar_plot.width_policy="min"
    bar_plot.min_width=50
    bar_plot.min_height=200
    bar_plot.height_policy="min"
    bar_plot.max_width=PLOT_WIDTH

    return bars, bar_plot, bar_view, bar_index, bar_source


def make_tables(summary_embeddings_df):

    # table_cols_list = ['summary', 'target', 'resolution', 'issue_id', 'bmu']
    table_cols_attrs = {
        'summary_original': {'width': 300},
        'target': {'width': 100},
        'issue_id': {'width': 70},
    }


    placeholder_style = dict({
                          'padding': '1em 1em 1em 0',
                          })



    table_cds = ColumnDataSource(summary_embeddings_df[['summary_original', 'target', 'resolution', 'issue_id', 'bmu']])
    table_cols = [ TableColumn(field=i, 
                            title=i, 
                            width=table_cols_attrs[i]['width']
                            ) for i in table_cols_attrs 
                ]

    table_index = IndexFilter(indices=[])
    table_all   = AllIndices()
    table_none  = IndexFilter(indices=[])
    table_group = GroupFilter(column_name="bmu")

    table_view  = CDSView(filter=table_all)
    node_table = DataTable(source=table_cds, 
                        columns=table_cols,
                        view=table_view,
                        index_position=-1
                        )

    node_stats_text = "Node Information Placeholder"
    node_stats = Div(text=node_stats_text, styles=placeholder_style)

    node_detail_text = "Click on  a row for details"
    node_detail = Div(text=node_detail_text, styles=placeholder_style)


    row_select_cb = CALLBACKS['row_select_cb']

    table_cds.selected.js_on_change('indices', 
                                    CustomJS(args=dict(
                                                table_src = table_cds,
                                                node_div = node_detail
                                                ),
                                            code=row_select_cb
                                            ))

    ## Table attributes
    node_table.sizing_mode="stretch_width"
    node_table.width_policy="min"
    node_table.min_width=50
    node_table.min_height=100
    node_table.height_policy="max"
    node_table.max_width=PLOT_WIDTH
    node_table.max_height=PLOT_HEIGHT


    node_detail.css_classes=["node_detail"]
    node_detail.height = 150
    node_detail.min_height = 100
    node_detail.height_policy='min'
    node_detail.max_width=PLOT_WIDTH

    return node_detail, node_table, table_cds, table_view

