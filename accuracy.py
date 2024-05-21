from itertools import product

import numpy as np
import sys
from munkres import Munkres

def clustering_error(predicted_biclustering, reference_biclustering, num_rows, num_cols):
    """The Clustering Error (CE) external evaluation measure.

    CE computes the similarity between two subspace clusterings. This measure was originally
    introduced in (Patrikainen and Meila, 2006) as a dissimilarity measure. In this package, it
    was implemented as a similarity measure as presented in (Horta and Campello, 2014). This measure
    lies in the interval [0, 1], where values close to 1 indicate better biclustering solutions.

    Reference
    ---------
    Patrikainen, A., & Meila, M. (2006). Comparing subspace clusterings. IEEE Transactions on
    Knowledge and Data Engineering, 18(7), 902-916.

    Horta, D., & Campello, R. J. G. B. (2014). Similarity measures for comparing biclusterings.
    IEEE/ACM Transactions on Computational Biology and Bioinformatics, 11(5), 942-954.

    Parameters
    ----------
    predicted_biclustering : biclustlib.model.Biclustering
        Predicted biclustering solution.

    reference_biclustering : biclustlib.model.Biclustering
        Reference biclustering solution.

    num_rows : int
        Number of rows of the dataset.

    num_cols : int
        Number of columns of the dataset.

    Returns
    -------
    ce : float
        Similarity score between 0.0 and 1.0.
    """
    # check = check_biclusterings(predicted_biclustering, reference_biclustering)

    # if isinstance(check, float):
    #     return check

    union_size = _calculate_size(predicted_biclustering, reference_biclustering, num_rows, num_cols, 'union')
    dmax = _calculate_dmax(predicted_biclustering, reference_biclustering)

    return float(dmax) / union_size

def _calculate_size(predicted_biclustering, reference_biclustering, num_rows, num_cols, operation):
    pred_count = _count_biclusters(predicted_biclustering, num_rows, num_cols)
    true_count = _count_biclusters(reference_biclustering, num_rows, num_cols)

    if operation == 'union':
        return np.sum(np.maximum(pred_count, true_count))
    elif operation == 'intersection':
        return np.sum(np.minimum(pred_count, true_count))

    valid_operations = ('union', 'intersection')

    raise ValueError("operation must be one of {0}, got {1}".format(valid_operations, operation))

def _calculate_dmax(predicted_biclustering, reference_biclustering):
    pred_sets = _bic2sets(predicted_biclustering)
    true_sets = _bic2sets(reference_biclustering)
    cost_matrix = [[sys.maxsize - len(b.intersection(g)) for g in true_sets] for b in pred_sets]
    indices = Munkres().compute(cost_matrix)
    return sum(sys.maxsize - cost_matrix[i][j] for i, j in indices)

def _count_biclusters(biclustering, num_rows, num_cols):
    count = np.zeros((num_rows, num_cols), dtype=int)

    for b in biclustering.biclusters:
        count[np.ix_(b.rows, b.cols)] += 1

    return count

def _bic2sets(biclust):
    return [set(product(b.rows, b.cols)) for b in biclust.biclusters]
