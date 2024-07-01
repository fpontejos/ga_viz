console.log(99999999)
console.log(cb_obj)

var selected  =  []
var indices  =  []

var is_visible = cb_obj.indices

for (var i = 0; i<geo_source.get_length(); i++) {
    indices.push(i)
    if (is_visible.includes(i)) {
        selected.push(i)
    }
}


// console.log('selected')
// console.log(selected)
// geo_index.indices  =  selected
// geo_view.filter  =  geo_index
// geo_source.selected.indices  =  selected
// geo_source.change.emit()
// splot.change.emit()
// console.log(splot)

// var geo_units  =  cb_obj.indices

// //var bmus  =  cb_obj.indices

// //console.log(geo_units)
// //console.log(geo_source)

// if (cb_obj.indices.length  ==  0) {

//     var indices  =  []
    
//     for (var i = 0; i<geo_source.get_length(); i++) {
//         indices.push(i)
//     }
    
//     //geo_source.selected.indices  =  []

//     //geo_index.indices  =  indices
//     //geo_view.filter  =  geo_index
    
//     //geo_source.change.emit()
    
    
// } else {
//     var selected  =  []
//     var indices  =  []
    

//     for (var i = 0; i<geo_source.get_length(); i++) {
//         indices.push(i)
        
//         var geo_unit  =  geo_source.data
//         //if (bmus.includes(geo_unit.BMU[i])) {
//         //    selected.push(i)
//         //}
//     }

    
//     //geo_index.indices  =  indices
//     //geo_view.filter  =  geo_index

//     //geo_source.selected.indices  =  selected
//     //geo_source.change.emit()
    
// }