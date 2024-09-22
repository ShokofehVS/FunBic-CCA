from itertools import product
from munkres import Munkres
from math import sqrt

import numpy as np
import sys

"""
    biclustlib: A Python library of biclustering algorithms and evaluation measures.
    Copyright (C) 2017  Victor Alexandre Padilha

    This file is part of biclustlib.

    biclustlib is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    biclustlib is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


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


def liu_wang_match_score(predicted_biclustering, reference_biclustering):
    """Liu & Wang match score.

    Reference
    ---------
    Liu, X., & Wang, L. (2006). Computing the maximum similarity bi-clusters of gene expression data.
    Bioinformatics, 23(1), 50-56.

    Horta, D., & Campello, R. J. G. B. (2014). Similarity measures for comparing biclusterings.
    IEEE/ACM Transactions on Computational Biology and Bioinformatics, 11(5), 942-954.

    Parameters
    ----------
    predicted_biclustering : biclustlib.model.Biclustering
        Predicted biclustering solution.

    reference_biclustering : biclustlib.model.Biclustering
        Reference biclustering solution.

    Returns
    -------
    lw_match_score : float
        Liu and Wang match score between 0.0 and 1.0.
    """

    k = len(predicted_biclustering.biclusters)

    return sum(max((len(np.intersect1d(bp.rows, br.rows)) + len(np.intersect1d(bp.cols, br.cols))) /
        (len(np.union1d(bp.rows, br.rows)) + len(np.union1d(bp.cols, br.cols)))
        for br in reference_biclustering.biclusters)
        for bp in predicted_biclustering.biclusters) / k


def prelic_relevance(predicted_biclustering, reference_biclustering):
    """The overall relevance match score defined in the supplementary material of Prelic et al. (2006).
    This score reflects how well the predicted biclusters represent the reference biclusters in both dimensions
    (rows and columns). This measure lies in the interval [0, 1], where values close to 1 indicate better
    biclustering solutions.

    Reference
    ---------
    Prelic, A., Bleuler, S., Zimmermann, P., Wille, A., Buhlmann, P., Gruissem, W., Hennig, L., Thiele, L. &
    Zitzler, E. (2006). A systematic comparison and evaluation of biclustering methods for gene expression data.
    Bioinformatics, 22(9), 1122-1129.

    Horta, D., & Campello, R. J. G. B. (2014). Similarity measures for comparing biclusterings.
    IEEE/ACM Transactions on Computational Biology and Bioinformatics, 11(5), 942-954.

    Parameters
    ----------
    predicted_biclustering : biclustlib.model.Biclustering
        Predicted biclustering solution.

    reference_biclustering : biclustlib.model.Biclustering
        Reference biclustering solution.

    Returns
    -------
    prel : float
        Similarity score between 0.0 and 1.0.
    """

    row_score = _match_score(predicted_biclustering, reference_biclustering, 'rows')
    col_score = _match_score(predicted_biclustering, reference_biclustering, 'cols')

    return sqrt(row_score * col_score)


def _match_score(predicted_biclustering, reference_biclustering, bicluster_attr):
    k = len(predicted_biclustering.biclusters)
    return sum(max(len(np.intersect1d(getattr(bp, bicluster_attr), getattr(bt, bicluster_attr))) /
        len(np.union1d(getattr(bp, bicluster_attr), getattr(bt, bicluster_attr)))
        for bt in reference_biclustering.biclusters)
        for bp in predicted_biclustering.biclusters) / k
