from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score


def get_cluster_metrics(som, df, vec, ltruecol, lpredcol):
    scores = {}
    labels_true = df[ltruecol]
    labels_pred = df[lpredcol]
    scores['homogeneity'] = homogeneity_score(labels_true, labels_pred)
    scores['completeness'] = completeness_score(labels_true, labels_pred)
    scores['v_measure'] = v_measure_score(labels_true, labels_pred)
    scores['quantization_err'] = som.quantization_error(vec)
    ## topographic error calc doesn't seem to work;
    ## throws index out of bounds error
    ## scores['topographic_err'] = som.topographic_error(vec)
    
    return scores