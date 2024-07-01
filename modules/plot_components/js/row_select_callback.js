var row_idx = cb_obj.indices[0]
var summary = table_src.data.summary_original[row_idx]

var data_ = {
'bmu': table_src.data.bmu[row_idx],
'issue_id': table_src.data.issue_id[row_idx],
'resolution': table_src.data.resolution[row_idx],
'target': table_src.data.target[row_idx],
'summary_original': table_src.data.summary_original[row_idx]
}

node_div.text = `
<strong>${data_['issue_id']}</strong>
<p>${data_['resolution']} by ${data_['target']}</p>
<br>
${data_['summary_original']}
`