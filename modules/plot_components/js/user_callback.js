var selected_user = cb_obj.value
    
var show_bmu_indices = []
for (var i = 0; i<user_nodes_src.get_length(); i++){
    if (user_nodes_src.data.target[i]==selected_user) {
        show_bmu_indices.push(user_nodes_src.data.bmu[i])
    }
}

var bmus = [...new Set(show_bmu_indices)]

user_index.indices = bmus
user_view.filter = user_index
user_src.change.emit()