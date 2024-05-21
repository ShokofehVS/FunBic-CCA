"""
    FunBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Functional Secret
    Sharing

    Copyright (C) 2024  Shokofeh VahidianSadegh

    This file is part of FunBic-CCA.

"""
from _base import BaseBiclusteringAlgorithm
from models import Bicluster, Biclustering
from sklearn.utils.validation import check_array

import numpy as np
import bottleneck as bn
import random
import math
import funshade


class ChengChurchAlgorithm(BaseBiclusteringAlgorithm):
    """Cheng and Church's Algorithm (CCA)

    CCA searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold.

    Reference
    ----------
    Cheng, Y., & Church, G. M. (2000). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    Parameters
    ----------
    num_biclusters : int, default: 100
        Number of biclusters to be found.

    msr_threshold : float or str, default: 'estimate'
        Maximum mean squared residue accepted (delta parameter in the original paper).
        If 'estimate', the algorithm will calculate this threshold as:
        (((max(data) - min(data)) ** 2) / 12) * 0.005.

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns (alpha parameter in the original paper).

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """

    def __init__(self, num_biclusters, msr_threshold, multiple_node_deletion_threshold, data_min_cols):
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold
        self.data_min_cols = data_min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """

        data = check_array(data, dtype=int, copy=True)
        self._validate_parameters()

        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)

        biclusters = []

        for i in range(self.num_biclusters):
            rows = np.ones(num_rows, dtype=bool)
            cols = np.ones(num_cols, dtype=bool)

            self._multiple_node_deletion(data, rows, cols, self.msr_threshold)
            self._node_addition(data, rows, cols)

            row_indices = np.nonzero(rows)[0]
            col_indices = np.nonzero(cols)[0]

            if len(row_indices) == 0 or len(col_indices) == 0:
                break

            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len(row_indices), len(col_indices))
                data[row_indices[:, np.newaxis], col_indices] = np.random.uniform(low=min_value, high=max_value,
                                                                                  size=bicluster_shape)

            biclusters.append(Bicluster(row_indices, col_indices))

        return Biclustering(biclusters)

    def _multiple_node_deletion(self, data, rows, cols, msr_thr):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described
        in the original paper)"""
        msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)
        stop = True if msr <= msr_thr else False

        while not stop:
            if len(rows) or len(cols) >= self.data_min_cols:
                cols_old = np.copy(cols)
                rows_old = np.copy(rows)

                fss_rs_rows = self.fss_evaluation(row_msr - self.multiple_node_deletion_threshold * msr)
                rows2remove = np.nonzero(fss_rs_rows)
                rows[rows2remove] = False

                msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

                fss_rs_cols = self.fss_evaluation(col_msr - self.multiple_node_deletion_threshold * msr)
                cols2remove = np.nonzero(fss_rs_cols)
                cols[cols2remove] = False

            msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

            # Tests if the new MSR value is smaller than the acceptable MSR threshold.
            # Tests if no rows and no columns were removed during this iteration.
            # If one of the conditions is true the loop must stop, otherwise it will become an infinite loop.
            if msr <= msr_thr or (np.all(rows == rows_old) and np.all(cols == cols_old)):
                stop = True

    def _node_addition(self, data, rows, cols):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
        the original paper)"""

        stop = False
        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            msr, _, _ = self._calculate_msr(data, rows, cols)
            col_msr = self._calculate_msr_col_addition(data, rows, cols)

            cols2add_fss = self.fss_evaluation(msr - col_msr)
            cols2add = np.nonzero(cols2add_fss)
            cols[cols2add] = True

            msr, _, _ = self._calculate_msr(data, rows, cols)
            row_msr = self._calculate_msr_row_addition(data, rows, cols)

            row2add_fss = self.fss_evaluation(msr - row_msr)
            rows2add = np.nonzero(row2add_fss)
            rows[rows2add] = True

            if np.all(rows == rows_old) and np.all(cols == cols_old):
                stop = True

    def _calculate_msr(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""
        sub_data = data[rows][:, cols]

        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data, axis=0)

        residues = sub_data - row_means[:, np.newaxis] - col_means + data_mean
        squared_residues = residues * residues

        msr = np.mean(squared_residues)
        row_msr = np.mean(squared_residues, axis=1)
        col_msr = np.mean(squared_residues, axis=0)

        return msr, row_msr, col_msr

    def _calculate_msr_col_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the columns for the node addition step."""
        sub_data = data[rows][:, cols]
        sub_data_rows = data[rows]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data_rows, axis=0)

        col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean
        col_squared_residues = col_residues * col_residues
        col_msr = np.mean(col_squared_residues, axis=0)

        return col_msr

    def _calculate_msr_row_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the rows and of the inverse of the rows for
        the node addition step."""
        sub_data = data[rows][:, cols]
        sub_data_cols = data[:, cols]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data_cols, axis=1)
        col_means = np.mean(sub_data, axis=0)

        row_residues = sub_data_cols - row_means[:, np.newaxis] - col_means + data_mean
        row_squared_residues = row_residues * row_residues
        row_msr = np.mean(row_squared_residues, axis=1)

        return row_msr

    def fss_evaluation(self, input):
        """Calculate the function secrete sharing of the secret share of input and output whether to remove/ add nodes
        from/ to matrix"""
        # Input parameters threshold, matrix, and length of matrix
        gamma = 0
        z = input.astype(funshade.DTYPE)
        K = len(z)

        # Create two computing parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Generate setup preprocessing materials
        r_in0, r_in1, k0, k1 = funshade.FssGenSign(K, gamma)

        P0.r_in_j = r_in0
        P1.r_in_j = r_in1
        P0.k_j = k0
        P1.k_j = k1

        # Secrete share the input
        rng = np.random.default_rng(seed=42)
        z_0 = rng.integers(np.iinfo(funshade.DTYPE).min,
                           np.iinfo(funshade.DTYPE).max, size=K, dtype=funshade.DTYPE)
        z_1 = z - z_0

        # Send the shares to the parties
        P0.z_j = z_0
        P1.z_j = z_1

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        # Evaluation with FSS IC gate
        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        # Construct the output of both parties
        o = P0.o_j + P1.o_j

        return o

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError("multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))

        if self.data_min_cols < 100:
            raise ValueError("data_min_cols must be >= 100, got {}".format(self.data_min_cols))


