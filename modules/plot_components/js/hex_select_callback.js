
var bmus = cb_obj.indices


/***
 * Trigger Edges selection on linked graph
 */
var coords = []
for(var i = 0; i < bmus.length; i++){
    coords.push(main_cds.data.coords[bmus[i]])
}

var coords_i = []
for(var i=0; i<coords.length; i++){
    var ci = coords[i]
    var ci_tmp = ci.slice(1,ci.length-1).replace(" ","").split(",")
    coords_i.push([parseInt(ci_tmp[0]),parseInt(ci_tmp[1])] )
}

var edge_idx_list = []

for(var i = 0; i < edge_cds.get_length(); i++){
    if(bmus.includes(edge_cds.data['start'][i]) || bmus.includes(edge_cds.data['end'][i])){
        edge_idx_list.push(i)
    }
}

edge_cds.selected.indices = edge_idx_list
edge_cds.change.emit()

/***
 * End edges selection
 */

if (cb_obj.indices.length == 0) {
    // no hex selected
    table_view.filter = table_all


    var bar_indices = []
    var new_bar_range = []
    
    for (var i = 0; i < bar_source.get_length(); i++) {
        if (bar_source.data['bmu'][i] == 'top') {
            bar_indices.push(i)
            new_bar_range.push(bar_source.data['target'][i])
        }
    }
    
    bar_index.indices = bar_indices
    bar_view.filter = bar_index
    
    bar_plot.title.text = 'Top 10 Users'
    
    bar_plot.x_range.factors = new_bar_range;
    bar_source.change.emit()
    
} else {
    var indices = []
    var node_data = []

    for (var i=0; i<table_src.get_length(); i++) {
        var table_row = table_src.data

        if (bmus.includes(table_row.bmu[i])) {
        
            var table_row_data = {
                'issue_id': table_src.data.issue_id[i],
                'resolution': table_src.data.resolution[i],
                'target': table_src.data.target[i]
            }

            node_data.push(table_row_data)
            indices.push(i)
        }
    }
    var bar_indices = []
    var new_bar_range = []
    
    for (var i = 0; i < bar_source.get_length(); i++) {
        if (bar_source.data['bmu'][i] == bmus[0].toString()) {
            bar_indices.push(i)
            new_bar_range.push(bar_source.data['target'][i])
        }
    }
    
    bar_index.indices = bar_indices
    bar_view.filter = bar_index
    
    var top_3 = (new_bar_range.slice(0,3))
    bar_plot.title.text = 'Top Users: ' + top_3.join(', ')
    
    bar_plot.x_range.factors = new_bar_range;
    bar_source.change.emit()
    

    table_index.indices = indices
    table_view.filter = table_index
    
    
}

table_src.change.emit()

