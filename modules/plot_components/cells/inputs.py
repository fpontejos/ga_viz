import numpy as np

from bokeh.models import ColumnDataSource, CDSView, CustomJS

from bokeh.models import Select, TextInput

from bokeh.models import IndexFilter


from modules.plot_components.js_callbacks import *
from constants import *

def make_user_input(summary_embeddings_df, m,n, x2,y2, ASSIGNEES):

    user_nodes = np.arange(m*n)

    user_src = ColumnDataSource(dict(
            x=x2,
            y=y2,
            users=user_nodes
        )
    )
    user_nodes_src = ColumnDataSource(summary_embeddings_df[['target', 'bmu']])


    user_view = CDSView(filter=IndexFilter(indices=[]))
    user_index = IndexFilter(indices=[])

    user_callback_code = CALLBACKS['user_callback_code']

    user_select = Select(value="All", 
                        options = ["All"] + ASSIGNEES, 
                        styles=dict({'width':'200px'}))
    user_select.js_on_change("value", 
                            CustomJS(args=dict(user_nodes_src=user_nodes_src,
                                                user_src = user_src,
                                                user_view = user_view,
                                                user_index = user_index
                                                ), 
                                    code=user_callback_code))


    return user_src, user_view, user_select

def make_query_input(transformer, codebook_tree, m,n, highlight_src, highlight_hex):

    query_input = TextInput(placeholder='Search Query')

    # Set up callbacks
    def highlight_winner_node(attrname, old, new):

        topnn = codebook_tree.query(transformer.encode(new), TOPNN_K)[1].tolist()
        highlight_rank = np.zeros(m*n) #+ .5
        
        rank_alpha = (np.arange(TOPNN_K,0,-1)+1)/(1+TOPNN_K)
        for i in range(TOPNN_K):
            highlight_rank[topnn[i]] = np.max([rank_alpha[i],0.4])

        
        highlight_src.data['rank'] = highlight_rank
        highlight_hex.view = CDSView(filter=IndexFilter(indices=topnn))

    query_input.on_change('value', highlight_winner_node)

    return query_input

