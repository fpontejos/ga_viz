import networkx as nx
import numpy as np


from bokeh.models import GraphRenderer
from bokeh.models import Circle, ColumnDataSource, MultiLine, NodesAndLinkedEdges, LabelSet
from bokeh.models import GraphRenderer, StaticLayoutProvider
from bokeh.models import ColumnDataSource
from bokeh.models import LinearColorMapper, LabelSet

from modules.plot_components.utils.interstitials import * 

from constants import * 

palette = 'Viridis256'



def make_graph(weights, hits):
    G = nx.Graph(seed=0, normalize=45)
    G = get_edges(weights, hits, G)
    Gpos = nx.nx_agraph.pygraphviz_layout(G, prog='neato', args='-Gmodel="mds" ')

    return G, Gpos

def get_edges(weights, hits_matrix, gx):
    
    im_m = (weights.shape[0]*2)-1
    im_n = (weights.shape[1]*2)-1
    
    interstitial_matrix = np.full((im_m, im_n), np.nan)
    interstitial_dirs = np.full((im_m, im_n), np.nan)
    s = weights.shape[:2]

    interstitial_max = 0
    
    hits_max = np.max(hits_matrix)
    
    
    def get_dist(node,neigh):
        dist = np.linalg.norm(weights[neigh]-weights[node])
        return dist
    def get_nodeid(i,j):
        return np.ravel_multi_index((i,j),(weights.shape[0],weights.shape[1]))
    
    
    for i in range(weights.shape[1]):
        for j in range(weights.shape[0]):
            radius = 1.5*hits_matrix[i][j]/hits_max
            #radius=.25

            if (i==0) and (j==0):
                gx.add_node(get_nodeid(i,j), 
                        width=radius, 
                        height=radius, 
                        label="0",
                        pos="0,0",
                        #pin="true"
                        #pos="1,1"
                       )
            elif (i==0) and (j==weights.shape[1]-1) :
                gx.add_node(get_nodeid(i,j), 
                        width=radius, 
                        height=radius, 
                        label="0,1",
                        #pos="0,1",
                        #pin="true"
                       )
            else:
                gx.add_node(get_nodeid(i,j), 
                        width=radius, 
                        height=radius, 
                        label=""
                       )
                
            
            neighs, inters_neighs, links_dir = get_neighbors(i,j,s)
            dist_tot = 0

            for n_i in range(inters_neighs.shape[0]) : # for each neighbor n_i,
                n_ii = inters_neighs[n_i][0]
                n_ij = inters_neighs[n_i][1]
                
                # if nan, this dist pair hasn't been calculated yet
                if np.isnan(interstitial_matrix[n_ii, n_ij]): # get distance from n_i to node at i,j
                    dist_i = get_dist( (i,j), (neighs[n_i][0], neighs[n_i][1]))
                    interstitial_matrix[n_ii, n_ij] = dist_i
                    interstitial_max = max(interstitial_max, dist_i)
                else:
                    dist_tot += interstitial_matrix[n_ii, n_ij]
            
            interstitial_matrix[i*2, j*2] = dist_tot/inters_neighs.shape[0]

    
    print(interstitial_max)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            
            neighs, inters_neighs, links_dir = get_neighbors(i,j,s, np.arange(3))

            for n_i in range(inters_neighs.shape[0]) : # for each neighbor n_i,
                n_ii = inters_neighs[n_i][0]
                n_ij = inters_neighs[n_i][1]

                w_i = np.round((interstitial_matrix[n_ii, n_ij]/interstitial_max), 2)*100
                # w_i = np.round(interstitial_matrix[n_ii, n_ij]/interstitial_max*5, 2)
                
                
                gx.add_edge(get_nodeid(i,j), 
                           get_nodeid(neighs[n_i][0], neighs[n_i][1]), 
                           dist=w_i,
                           len=int(w_i/25),
                        #    len=int(w_i),
                           #len=1,
                           label="{}".format(int(w_i/20))
                           )
                    
            
                
    return gx

def get_midpt(Gpos, p1, p2):
    p1=Gpos[p1]
    p2=Gpos[p2]

    return np.round(np.mean([p1[0],p2[0]]),2), np.round(np.mean([p1[1],p2[1]]),2)

def make_graph_plot(G, Gpos, plot, node_cds, cmap):

    # Flip because the graph gets plotted mirrored along diagonal axis
    flipped_pos = {node: (y,x) for (node, (x,y)) in Gpos.items()}

    # list the nodes and initialize a plot
    N = len(flipped_pos)

    
    labels_pos = [get_midpt(flipped_pos, i[0],i[1]) for i in G.edges]

    edge_dist = np.array([G.get_edge_data(i[0],i[1])['dist'] for i in G.edges])
    max_dist = np.max(edge_dist)
    min_dist = np.min(edge_dist)
    inv_dist = 1-((edge_dist-min_dist)/(max_dist-min_dist)) #edge_dist/np.max(edge_dist)
    edge_color = np.round(np.round(inv_dist*1,2),1)
    
    edge_cmap = LinearColorMapper(palette="Viridis256",
                             low = max(edge_color), 
                             high = min(edge_color))

    edge_cds = ColumnDataSource(data = dict(
        start   = [ i[0] for i in G.edges ],
        end     = [ i[1] for i in G.edges ],
        edge_dist =   edge_dist,
        dist    = [G.get_edge_data(i[0],i[1])['dist'] for i in G.edges],
        mdpt_x  = [i[0] for i in labels_pos],
        mdpt_y  = [i[1] for i in labels_pos],
        inv_dist = inv_dist,
        
        edge_width = np.round(10*np.round(inv_dist*.4,2),0),
        edge_alpha = np.round(np.round(inv_dist*.4,2),1),
        edge_color = edge_color
        
    ))

    labels_cds = ColumnDataSource(data=dict(x=[], y=[], dist=[]))
    labels_empty = ColumnDataSource(data=dict(x=[], y=[], dist=[]))


    graph = GraphRenderer()


    graph.node_renderer.glyph = Circle(size='radius', #fill_color='#BBB')
                    fill_color={"field":"umat_flat", "transform":cmap},
                    line_color={"field":"umat_flat", "transform":cmap},
                    line_width=2,
                    fill_alpha=1,
                    )

    graph.node_renderer.hover_glyph = Circle(size='radius', 
                                            fill_color=CONTRAST_COLOR1, 
                                            line_color=CONTRAST_COLOR1, 
                                            line_width=4)

    graph.node_renderer.selection_glyph = Circle(size='radius', 
                    fill_color={"field":"umat_flat", "transform":cmap},
                    line_color={"field":"umat_flat", "transform":cmap},
                                            line_width=4)

    graph.node_renderer.nonselection_glyph = Circle(
                    fill_color="#AAA",
                    line_color="#AAA",
                    line_width=1,
                    fill_alpha=1)

    graph.edge_renderer.glyph = MultiLine(
                    line_color={"field":"edge_color", "transform":edge_cmap},
                    line_width='edge_width', 
                    line_alpha='edge_alpha'
                    )
    graph.edge_renderer.selection_glyph = MultiLine(
                    line_color={"field":"edge_color", "transform":edge_cmap},
                    line_width='edge_width', 
                    line_alpha=1
                    )
    graph.edge_renderer.nonselection_glyph = MultiLine(line_color='#DDD',
                    line_width='edge_width', 
                    line_alpha=.25
                    )
    graph.edge_renderer.hover_glyph = MultiLine(
                    line_color={"field":"edge_color", "transform":edge_cmap},
                    line_width='edge_width', 
                    line_alpha=1
                    )

    # assign a palette to ``fill_color`` and add it to the data source
    graph.node_renderer.data_source = node_cds

    # add the rest of the assigned values to the data source
    graph.edge_renderer.data_source = edge_cds

    # use the provider model to supply coordinates to the graph
    graph.layout_provider = StaticLayoutProvider(graph_layout=flipped_pos)


    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()


    # render the graph
    plot.renderers.append(graph)
    plot.toolbar.logo = None
    labels_cds = ColumnDataSource(data=dict(x=[], y=[], dist=[]))
    labels_empty = ColumnDataSource(data=dict(x=[], y=[], dist=[]))

    labels = LabelSet(x='mdpt_x',
                    y='mdpt_y',
                    text='edge_alpha', 
                    text_font_size="12px",
                    x_offset=2, 
                    y_offset=2,
                    source=edge_cds
                    )

    plot.grid.grid_line_color = None



    return plot, edge_cds
