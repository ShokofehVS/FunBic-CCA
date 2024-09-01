"""
    FunBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Functional Secret
    Sharing

    Copyright (C) 2024  Shokofeh VahidianSadegh

    This file is part of FunBic-CCA.

"""
from scipy.signal import residue

from _base import BaseBiclusteringAlgorithm
from models import Bicluster, Biclustering
from sklearn.utils.validation import check_array

import numpy as np
import funshade
import time
import os
import math


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

    msr_threshold : float, default: 300 or 1200
        Maximum mean squared residue accepted (delta parameter in the original paper).

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns (alpha parameter in the original paper).

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """

    def __init__(self, high, num_biclusters, msr_threshold, multiple_node_deletion_threshold, data_min_cols):
        self.highest_range = high
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

        # Create parties
        class party:
            def __init__(self, d: int):
                self.d = d

        P_0 = party(0)
        P_1 = party(1)

        # Definitions
        biclusters = []
        # t_share, t_score, t_comp = [], [], []
        t_shareMSR, t_shareEval, t_size = [], [], []

        # For number of biclusters
        for i in range(self.num_biclusters):
            # Shape of data
            num_rows, num_cols = data.shape
            min_value = np.min(data)
            max_value = np.max(data)

            # Generate secret shares of the input data
            rng = np.random.default_rng(seed=42)
            in_0 = rng.integers(1, self.highest_range, size=(num_rows, num_cols), dtype="int64")
            in_1 = data - in_0

            # Assign secret shares to parties
            P_0.aij_0 = np.copy(in_0)
            P_1.aij_1 = np.copy(in_1)

            # Shape of inputs for both parties
            num_row_0, num_col_0 = in_0.shape
            num_row_1, num_col_1 = in_1.shape

            # Input matrices before score finding and removal/ addition of nodes
            rows_0 = np.ones(num_row_0, dtype=bool)
            cols_0 = np.ones(num_col_0, dtype=bool)

            rows_1 = np.ones(num_row_1, dtype=bool)
            cols_1 = np.ones(num_col_1, dtype=bool)

            with open('result_size.txt', 'w') as saveFile:
                # saveFile.write("\n")
                saveFile.write(str(P_0.aij_0) + "\n")
                saveFile.write(str(P_1.aij_1) + "\n")
            t_size.append(os.path.getsize("result_size.txt"))

            # Steps including multiple deletion/ addition
            P_0.bij_0, P_1.bij_1, fss_rs_rows_0, fss_rs_rows_1 = self._multiple_node_deletion(P_0.aij_0, P_1.aij_1,
                                                                                              rows_0, cols_0,
                                                                                              rows_1, cols_1,
                                                                                              self.msr_threshold, t_shareMSR, t_shareEval)
            P_0.cij_0, P_1.cij_1 = self._node_addition(P_0.bij_0, P_1.bij_1, in_0, in_1,
                                                       rows_0, cols_0,
                                                       rows_1, cols_1,
                                                       fss_rs_rows_0, fss_rs_rows_1, t_shareMSR, t_shareEval)



            # Output shares to reconstruct the final matrix
            new_data = P_0.cij_0 + P_1.cij_1

            # Number of rows and columns
            new_cols             = new_data.shape[1]
            rows_without_zeros   = ~np.any(new_data == 0, axis=1)
            cols_without_zeros   = ~np.any(new_data == 0, axis=0)
            sub_new_data         = new_data[rows_without_zeros]
            new_rows             = sub_new_data.shape[0]

            # Rows and columns indexes
            rows_indexes  = np.where(rows_without_zeros)[0]
            cols_indexes  = np.where(cols_without_zeros)[0]


            if new_rows == 0 or new_cols == 0:
                break

            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (new_rows, new_cols)
                data            = np.random.uniform(low=min_value, high=max_value, size=bicluster_shape)

            biclusters.append(Bicluster(rows_indexes, cols_indexes))

        # Write the performance time
        with open('yeast_performance.txt', 'a') as saveFile:
            saveFile.write("\n")
            # saveFile.write("Performance of MSR:"  + str(np.mean(t_shareMSR)) + "\n")
            # saveFile.write("Performance of Eval:" + str(np.mean(t_shareEval)) + "\n")
            saveFile.write("Number of bits     :" + str(self.highest_range) + "\n")
            saveFile.write("Communication size :" + str(np.mean(t_size)) + "\n")
            # saveFile.write("\n")


        return Biclustering(biclusters)


    def _multiple_node_deletion(self, P_in_0, P_in_1, rows_0, cols_0, rows_1, cols_1, msr_thr, t_shareMSR, t_shareEval):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described
        in the original paper)"""
        # Secret shared inputs
        in_0 = P_in_0
        in_1 = P_in_1

        # Secret shared inputs' shapes
        num_row_0, num_col_0 = P_in_0.shape
        num_row_1, num_col_1 = P_in_1.shape

        # MSRs computation when any nodes are removed (having exact rows/ columns length)
        t_hs_0 = time.perf_counter()
        msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1  = self._scores_before_steps(in_0, in_1)
        t_hs_1 = time.perf_counter()
        t_shareMSR.append(t_hs_1 - t_hs_0)


        # Check whether the MSR is below or equal to threshold (leak info after fss gate)
        stop_itr_0 = msr_thr - msr_0
        stop_itr_1 = msr_thr - msr_1
        stop       = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

        # Inner stop function (fixed amount of iterations) in case of having columns below 100
        stop_con2 = math.log(num_row_0)

        if stop:
            fss_rs_rows_0 = np.zeros(num_row_0)
            fss_rs_rows_1 = np.zeros(num_row_1)
            return in_0, in_1, fss_rs_rows_0, fss_rs_rows_1

        else:
            while not stop:

                for itr in range(round(stop_con2)):
                    # FSS gate to check which rows should be removed
                    # Calculate the performance and communication of Eval
                    t_sh_0 = time.perf_counter()
                    r2remove_con_0 = self.multiple_node_deletion_threshold * msr_0 - row_msr_0
                    r2remove_con_1 = self.multiple_node_deletion_threshold * msr_1 - row_msr_1
                    fss_rs_rows_0, fss_rs_rows_1 = self.fss_evaluation_without_len(r2remove_con_0, r2remove_con_1)

                    # Create a matrix for FSS results before multiplication of them with input matrix
                    fss_rs_rows_tile_0 = np.tile(fss_rs_rows_0, (num_col_0, 1)).T
                    fss_rs_rows_tile_1 = np.tile(fss_rs_rows_1, (num_col_1, 1)).T

                    # Now mask rows with zeros to be removed for those in FSS gate result
                    for idxr in range(num_row_0):
                        in_0[idxr], in_1[idxr] = self.secMult_vector(in_0[idxr], in_1[idxr],
                                                                     fss_rs_rows_tile_0[idxr], fss_rs_rows_tile_1[idxr])

                    t_sh_1 = time.perf_counter()
                    t_shareEval.append(t_sh_1 - t_sh_0)

                    # Check whether columns are above 100
                    if len(cols_0) >= self.data_min_cols:
                        in_0, in_1 = self._cols2Remove(in_0, in_1, rows_0, rows_1, cols_0, cols_1,
                                                       num_row_0, num_col_0, num_row_1, num_col_1)

                    # Recalculate the scores (NOTE the columns by default have not been removed)
                    t_hs_0 = time.perf_counter()
                    msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1 = self._calculate_scores_multidel(in_0, in_1,
                                                                            fss_rs_rows_0, fss_rs_rows_1,
                                                                            num_col_0, num_col_1)
                    t_hs_1 = time.perf_counter()
                    t_shareMSR.append(t_hs_1 - t_hs_0)

                # Check whether the MSR is below/equal to threshold
                stop_itr_0 = msr_thr - msr_0
                stop_itr_1 = msr_thr - msr_1
                stop = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

            return in_0, in_1, fss_rs_rows_0, fss_rs_rows_1


    def _node_addition(self, bij_0, bij_1, in_0, in_1, rows_0, cols_0, rows_1, cols_1, fss_rs_rows_0, fss_rs_rows_1, t_shareMSR, t_shareEval):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
        the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape
        num_row_1, num_col_1 = in_1.shape

        # Following steps are only be applicable in situation having columns more than 100 (|J| >= 100)
        if len(cols_0) >= self.data_min_cols:
            bij_0, bij_1 = self._cols2Add(bij_0, bij_1, in_0, in_1,
                                          rows_0, rows_1, cols_0, cols_1,
                                          num_row_0, num_col_0, num_row_1, num_col_1)


        # Calculate score for whole matrix and that of rows
        t_hs_0 = time.perf_counter()
        msr_0, msr_1, row_msr_0, row_msr_1, _, _ = self._calculate_scores_multidel(bij_0, bij_1,
                                                                   fss_rs_rows_0, fss_rs_rows_1,
                                                                   num_col_0, num_col_1)
        t_hs_1 = time.perf_counter()
        t_shareMSR.append(t_hs_1 - t_hs_0)

        # Check whether rows are ready to add
        t_sh_0 = time.perf_counter()
        r2add_con_0 = msr_0 - row_msr_0
        r2add_con_1 = msr_1 - row_msr_1
        fss_rs_rows_add_0, fss_rs_rows_add_1 = self.fss_evaluation_without_len(r2add_con_0, r2add_con_1)

        # Then add those which are in the range of addition
        # Create a matrix for FSS results before multiplication of them with input matrix
        fss_rs_rows_add_0 = np.tile(fss_rs_rows_add_0, (num_col_0, 1)).T
        fss_rs_rows_add_1 = np.tile(fss_rs_rows_add_1, (num_col_1, 1)).T

        # Multiply by source input matrix and then add with the output of multiple deletion matrix
        tmp_bij_0 = np.zeros(bij_0.shape, dtype="int64");           tmp_bij_1 = np.zeros(bij_1.shape, dtype="int64")
        for idxr in range(num_row_0):
            tmp_bij_0[idxr], tmp_bij_1[idxr] = self.secMult_vector(in_0[idxr], in_1[idxr],
                                                         fss_rs_rows_add_0[idxr], fss_rs_rows_add_1[idxr])
        bij_0 += tmp_bij_0
        bij_1 += tmp_bij_1

        t_sh_1 = time.perf_counter()
        t_shareEval.append(t_sh_1 - t_sh_0)

        return bij_0, bij_1


    def _scores_before_steps(self, in_0, in_1):
        """Calculate scores of the rows, of the columns and of the full data matrix."""
        # Note that all mean are converted to integers to avoid overflow
        # Mean values
        data_mean_0 = np.mean(in_0).astype(int);                        data_mean_1 = np.mean(in_1).astype(int)
        row_means_0 = np.mean(in_0, axis=1).astype(int);                row_means_1 = np.mean(in_1, axis=1).astype(int)
        col_means_0 = np.mean(in_0, axis=0).astype(int);                col_means_1 = np.mean(in_1, axis=0).astype(int)

        # Residue
        residues_0 = (in_0 - row_means_0[:, np.newaxis] - col_means_0 + data_mean_0)
        residues_1 = (in_1 - row_means_1[:, np.newaxis] - col_means_1 + data_mean_1)

        # Continue doing squaring by local power of 2 residue and joint multiplication
        squared_residue_0 = np.copy(residues_0)
        squared_residue_1 = np.copy(residues_1)

        for idxr in range(residues_0.shape[0]):
            squared_residue_0[idxr], squared_residue_1[idxr] = self.secSquare_vector(residues_0[idxr],
                                                                                     residues_1[idxr])

        # MSRs computations
        msr_0     = np.mean(squared_residue_0).astype(int)
        msr_1     = np.mean(squared_residue_1).astype(int)

        row_msr_0 = np.mean(squared_residue_0, axis=1).astype(int)
        row_msr_1 = np.mean(squared_residue_1, axis=1).astype(int)

        col_msr_0 = np.mean(squared_residue_0, axis=0).astype(int)
        col_msr_1 = np.mean(squared_residue_1, axis=0).astype(int)

        return msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1


    def _calculate_scores_multidel(self, in_0, in_1, fss_rs_row_0, fss_rs_row_1, num_col_0, num_col_1):
        """Calculate residue only when rows are removed."""
        # Size of rows that are not removed
        sum_fss_rs_rows_0 = np.sum(fss_rs_row_0)
        sum_fss_rs_rows_1 = np.sum(fss_rs_row_1)

        # Convert to array
        sum_fss_rs_rows_0 = np.full(num_col_0, sum_fss_rs_rows_0)
        sum_fss_rs_rows_1 = np.full(num_col_1, sum_fss_rs_rows_1)

        # Generate setup preprocessing materials
        rng      = np.random.default_rng(seed=42)
        r_beta_0 = rng.integers(1, self.highest_range, size=num_col_0, dtype="int64")
        r_beta   = rng.integers(1, self.highest_range, size=num_col_0, dtype="int64")
        r_beta_1 = r_beta - r_beta_0

        # Multiply the mask with sum of fss for rows
        mult_res_0, mult_res_1 = self.secMult_vector(sum_fss_rs_rows_0, sum_fss_rs_rows_1, r_beta_0, r_beta_1)

        # Reconstruct the result
        mult_res = mult_res_0 + mult_res_1

        # Mean values
        # Convert to array
        whole_sum_0 = np.sum(in_0)
        whole_sum_1 = np.sum(in_1)

        whole_sum_0 = np.full(num_col_0, whole_sum_0)
        whole_sum_1 = np.full(num_col_1, whole_sum_1)

        data_mean_tmp_0, data_mean_tmp_1 = self.secMult_vector(whole_sum_0, whole_sum_1, r_beta_0, r_beta_1)

        data_mean_0 = (data_mean_tmp_0 / (mult_res * num_col_0)).astype(int)
        data_mean_1 = (data_mean_tmp_1 / (mult_res * num_col_1)).astype(int)

        row_mean_0 = (np.sum(in_0, axis=1) / num_col_0).astype(int)
        row_mean_1 = (np.sum(in_1, axis=1) / num_col_1).astype(int)

        col_mean_tmp_0, col_mean_tmp_1 = self.secMult_vector(np.sum(in_0, axis=0), np.sum(in_1, axis=0),
                                                             r_beta_0, r_beta_1)

        col_mean_0 = (col_mean_tmp_0 / (mult_res)).astype(int)
        col_mean_1 = (col_mean_tmp_1 / (mult_res)).astype(int)

        # Residue
        residue_0 = in_0 - row_mean_0[:, np.newaxis] - col_mean_0 + data_mean_0[0]
        residue_1 = in_1 - row_mean_1[:, np.newaxis] - col_mean_1 + data_mean_1[0]

        # Continue doing squaring by local power of 2 residue and joint multiplication
        squared_residue_0 = np.copy(residue_0)
        squared_residue_1 = np.copy(residue_1)

        for idxr in range(residue_0.shape[0]):
            squared_residue_0[idxr], squared_residue_1[idxr] = self.secSquare_vector(residue_0[idxr],
                                                                                     residue_1[idxr])

        # Mean squared residues
        msr_0 = (np.mean(squared_residue_0)).astype(int)
        msr_1 = (np.mean(squared_residue_1)).astype(int)

        row_msr_0 = (np.mean(squared_residue_0, axis=1)).astype(int)
        row_msr_1 = (np.mean(squared_residue_1, axis=1)).astype(int)

        col_msr_0 = (np.mean(squared_residue_0, axis=0)).astype(int)
        col_msr_1 = (np.mean(squared_residue_1, axis=0)).astype(int)


        return msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1


    def _cols2Remove(self, in_0, in_1, rows_0, rows_1, cols_0, cols_1, num_row_0, num_col_0, num_row_1, num_col_1):
        # Recalculate the score
        # Calculate First residue locally
        residue_0 = self._calculate_residue(in_0, rows_0, cols_0, 1, 0, 0)
        residue_1 = self._calculate_residue(in_1, rows_1, cols_1, 1, 0, 0)

        # Utilise secured vector multiplication with one round of communication for r_ij_0 * r_ij_1
        residue_0_flatted = residue_0.flatten()
        residue_1_flatted = residue_1.flatten()

        squared_residue_0_flatted, squared_residue_1_flatted = self.secMult_vector(residue_0_flatted,
                                                                                   residue_1_flatted,
                                                                                   residue_0_flatted,
                                                                                   residue_1_flatted)

        # Continue doing squaring by local power of 2 residue and joint multiplication
        squared_residue_0 = residue_0 ** 2 + (2 * squared_residue_0_flatted.reshape(num_row_0, num_col_0))
        squared_residue_1 = residue_1 ** 2 + (2 * squared_residue_1_flatted.reshape(num_row_1, num_col_1))

        # Local MSRs computation
        msr_0, row_msr_0, col_msr_0 = self._calculate_msr(squared_residue_0, 1, 0, 0)
        msr_1, row_msr_1, col_msr_1 = self._calculate_msr(squared_residue_1, 1, 0, 0)

        # FSS gate to check which cols should be removed
        c2remove_con_0 = self.multiple_node_deletion_threshold * msr_0 - col_msr_0
        c2remove_con_1 = self.multiple_node_deletion_threshold * msr_1 - col_msr_1
        fss_rs_cols_0, fss_rs_cols_1 = self.fss_evaluation_without_len(c2remove_con_0, c2remove_con_1)

        # Create a matrix for FSS results before multiplication of them with input matrix
        fss_rs_cols_0 = np.tile(fss_rs_cols_0, (num_row_0, 1)).T
        fss_rs_cols_1 = np.tile(fss_rs_cols_1, (num_row_1, 1)).T

        # Now mask cols with zeros to be removed for those in FSS gate result
        # First transpose secret shared matrices before multiplying with fss
        transposed_in_0 = in_0.T
        transposed_in_1 = in_1.T
        # Secondly multiplying transposed matrices with fss results
        for idxc in range(num_col_0):
            transposed_in_0[idxc], transposed_in_1[idxc] = self.secMult_vector(
                transposed_in_0[idxc], transposed_in_1[idxc],
                fss_rs_cols_0[idxc], fss_rs_cols_1[idxc])
        # Thirdly return the transposed matrices to normal
        in_0 = transposed_in_0.T
        in_1 = transposed_in_1.T

        return in_0, in_1


    def _cols2Add(self, bij_0, bij_1, in_0, in_1,
                                          rows_0, rows_1, cols_0, cols_1,
                                          num_row_0, num_col_0, num_row_1, num_col_1):
        # Find out the MSR for whole matrix like in multiple node deletion
        # Calculate First residue locally
        residue_0 = self._calculate_residue(in_0, rows_0, cols_0, 1, 0, 0)
        residue_1 = self._calculate_residue(in_1, rows_1, cols_1, 1, 0, 0)

        # Utilise secured vector multiplication with one round of communication for r_ij_0 * r_ij_1
        residue_0_flatted = residue_0.flatten()
        residue_1_flatted = residue_1.flatten()

        squared_residue_0_flatted, squared_residue_1_flatted = self.secMult_vector(residue_0_flatted, residue_1_flatted,
                                                                                   residue_0_flatted, residue_1_flatted)

        # Continue doing squaring by local power of 2 residue and joint multiplication
        squared_residue_0 = residue_0 ** 2 + (2 * squared_residue_0_flatted.reshape(num_row_0, num_col_0))
        squared_residue_1 = residue_1 ** 2 + (2 * squared_residue_1_flatted.reshape(num_row_1, num_col_1))

        # Local MSRs computation
        msr_0, _, _ = self._calculate_msr(squared_residue_0, 1, 0, 0)
        msr_1, _, _ = self._calculate_msr(squared_residue_1, 1, 0, 0)

        # Find out the MSR for columns
        # Calculate First residue locally
        residue_0 = self._calculate_residue(in_0, rows_0, cols_0, 0, 1, 0)
        residue_1 = self._calculate_residue(in_1, rows_1, cols_1, 0, 1, 0)

        # Utilise secured vector multiplication with one round of communication for r_ij_0 * r_ij_1
        residue_0_flatted = residue_0.flatten()
        residue_1_flatted = residue_1.flatten()

        squared_residue_0_flatted, squared_residue_1_flatted = self.secMult_vector(residue_0_flatted, residue_1_flatted,
                                                                                   residue_0_flatted, residue_1_flatted)

        # Continue doing squaring by local power of 2 residue and joint multiplication
        squared_residue_0 = residue_0 ** 2 + (2 * squared_residue_0_flatted.reshape(num_row_0, num_col_0))
        squared_residue_1 = residue_1 ** 2 + (2 * squared_residue_1_flatted.reshape(num_row_1, num_col_1))

        # Local MSR computation
        col_msr_0 = self._calculate_msr(squared_residue_0, 0, 1, 0)
        col_msr_1 = self._calculate_msr(squared_residue_1, 0, 1, 0)

        # FSS gate to check which cols should be added
        c2add_con_0 = msr_0 - col_msr_0
        c2add_con_1 = msr_1 - col_msr_1
        fss_rs_cols_0, fss_rs_cols_1 = self.fss_evaluation_without_len(c2add_con_0, c2add_con_1)

        # Then add those which are in the range of addition
        # Create a matrix for FSS results before multiplication of them with input matrix
        fss_rs_cols_0 = np.tile(fss_rs_cols_0, (num_row_0, 1)).T
        fss_rs_cols_1 = np.tile(fss_rs_cols_1, (num_row_1, 1)).T

        # Transpose secret shared matrices before multiplying with fss
        transposed_bij_0 = bij_0.T;
        transposed_tmp_bij_0 = np.zeros(transposed_bij_0.shape, dtype="int64")
        transposed_bij_1 = bij_1.T;
        transposed_tmp_bij_1 = np.zeros(transposed_bij_1.shape, dtype="int64")
        transposed_in_0 = in_0.T
        transposed_in_1 = in_1.T

        # Secondly multiplying transposed matrices with fss results
        for idxc in range(num_col_0):
            transposed_tmp_bij_0[idxc], transposed_tmp_bij_1[idxc] = self.secMult_vector(
                transposed_in_0[idxc], transposed_in_1[idxc],
                fss_rs_cols_0[idxc], fss_rs_cols_1[idxc])

        transposed_bij_0 += transposed_tmp_bij_0
        transposed_bij_1 += transposed_tmp_bij_1

        # Thirdly return the transposed matrices to normal
        bij_0 = transposed_bij_0.T
        bij_1 = transposed_bij_1.T

        return bij_0, bij_1


    def _calculate_residue(self, data, rows, cols, multidel, coladd, rowadd):
        """Calculate only residue locally of the rows, of the columns and of the full data matrix."""
        # Calculate performance
        # t_sc_0 = time.perf_counter()

        sub_data = data
        # Check which action is being performed; then compute local addition/ subtraction plus division
        # Note that all mean are converted to integers to avoid overflow
        if multidel:
            data_mean = np.mean(sub_data).astype(int)
            row_means = np.mean(sub_data, axis=1).astype(int)
            col_means = np.mean(sub_data, axis=0).astype(int)

            residues = (sub_data - row_means[:, np.newaxis] - col_means + data_mean)

            return residues

        elif coladd:
            sub_data = data[rows][:, cols]
            sub_data_rows = data[rows]
            data_mean = np.mean(sub_data).astype(int)
            row_means = np.mean(sub_data, axis=1).astype(int)
            col_means = np.mean(sub_data_rows, axis=0).astype(int)
            col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean

            return col_residues

        else:
            sub_data = data[rows][:, cols]
            sub_data_cols = data[:, cols]
            data_mean = np.mean(sub_data).astype(int)
            row_means = np.mean(sub_data_cols, axis=1).astype(int)
            col_means = np.mean(sub_data, axis=0).astype(int)
            row_residues = sub_data_cols - row_means[:, np.newaxis] - col_means + data_mean

            return row_residues


    def _calculate_msr(self, sq_residues, multidel, coladd, rowadd):
        """Calculate only MSRs locally of the rows, of the columns and of the full data matrix."""
        # Calculate performance
        """
        t_sc_0 = time.perf_counter()
        t_sc_1 = time.perf_counter()
        t_score.append(t_sc_1 - t_sc_0)

        with open('t_score_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(msr) + "\n")
            saveFile.write(str(row_msr) + "\n")
            saveFile.write(str(col_msr) + "\n")
            # saveFile.write("\n")

        t_sh_1 = time.perf_counter()
        t_share.append(t_sh_1 - t_sh_0)
        t_score.append(os.path.getsize("t_score_size.txt"))
"""
        # Check which action is being performed; then send msr, or that of rows and columns back
        if multidel:
            squared_residues = sq_residues
            msr = np.mean(squared_residues).astype(int)
            row_msr = np.mean(squared_residues, axis=1).astype(int)
            col_msr = np.mean(squared_residues, axis=0).astype(int)

            return msr, row_msr, col_msr

        elif coladd:
            col_squared_residues = sq_residues
            col_msr = np.mean(col_squared_residues, axis=0).astype(int)
            return col_msr

        else:
            row_squared_residues = sq_residues
            row_msr = np.mean(row_squared_residues, axis=1).astype(int)
            return row_msr


    def secSquare_vector(self, share_00, share_01):
        """Generation of Beaver Triples and secured multiplication for vectors."""
        # Input parameters including vectors, threshold, and length of matrix
        theta = 0
        z_0 = share_00.astype(funshade.DTYPE)
        z_1 = share_01.astype(funshade.DTYPE)

        K = len(z_0)

        # Create parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Distribute secret share values
        P0.z_j = z_0;                       P1.z_j = z_1

        # Generate beaver triples for vectors
        a_hat_0, a_hat_1, b_hat_0, b_hat_1 = funshade.beaverTriple_square(K, theta)
        # # Distribute randomness to (P0, P1)
        P0.hat_a_j = a_hat_0;               P1.hat_a_j = a_hat_1
        P0.hat_b_j = b_hat_0;               P1.hat_b_j = b_hat_1

        # Find e
        P0.hat_e_j = funshade.share_square(K, P0.z_j, P0.hat_a_j)
        P1.hat_e_j = funshade.share_square(K, P1.z_j, P1.hat_a_j)

        # Reconstruct e in one round of communication
        P0.e_j = P0.hat_e_j + P1.hat_e_j
        P1.e_j = P0.hat_e_j + P1.hat_e_j

        # Now square with beaver triples
        P0.sq_j = funshade.square(K, P0.j, P0.e_j, P0.hat_a_j, P0.hat_b_j)
        P1.sq_j = funshade.square(K, P1.j, P1.e_j, P1.hat_a_j, P1.hat_b_j)


        return P0.sq_j, P1.sq_j



    def secMult_vector(self, share_00, share_01, share_10, share_11):
        """Generation of Beaver Triples and secured multiplication for vectors."""
        # Input parameters including vectors, threshold, and length of matrix
        theta = 0
        z_0 = share_00.astype(funshade.DTYPE)
        z_1 = share_01.astype(funshade.DTYPE)

        y_0 = share_10.astype(funshade.DTYPE)
        y_1 = share_11.astype(funshade.DTYPE)

        K = len(z_0)

        # Create parties
        class party:
            def __init__(self, j: int):
                self.j = j

        P0 = party(0)
        P1 = party(1)

        # Generate beaver triples for vectors
        a_hat_0, a_hat_1, b_hat_0, b_hat_1, c_hat_0, c_hat_1 = funshade.beaverTriple_ss(K, theta)

        # Distribute randomness to (P0, P1)
        P0.hat_a_j = a_hat_0;           P1.hat_a_j = a_hat_1
        P0.hat_b_j = b_hat_0;           P1.hat_b_j = b_hat_1
        P0.hat_c_j = c_hat_0;           P1.hat_c_j = c_hat_1

        # Find d and e
        P0.hat_d_j = funshade.share_ss(K, z_0, P0.hat_a_j)
        P0.hat_e_j = funshade.share_ss(K, y_0, P0.hat_b_j)

        P1.hat_d_j = funshade.share_ss(K, z_1, P1.hat_a_j)
        P1.hat_e_j = funshade.share_ss(K, y_1, P1.hat_b_j)

        # Reconstruct d and e in one round of communication
        P0.hat_d = P0.hat_d_j + P1.hat_d_j
        P1.hat_d = P0.hat_d_j + P1.hat_d_j

        P0.hat_e = P0.hat_e_j + P1.hat_e_j
        P1.hat_e = P0.hat_e_j + P1.hat_e_j

        # Now multiply with additive secret sharing
        P0.node2del_j = funshade.node2del(K, P0.j, P0.hat_d, P0.hat_e, P0.hat_a_j, P0.hat_b_j, P0.hat_c_j)
        P1.node2del_j = funshade.node2del(K, P1.j, P1.hat_d, P1.hat_e, P1.hat_a_j, P1.hat_b_j, P1.hat_c_j)

        return P0.node2del_j, P1.node2del_j

    def fss_evaluation(self, share_0, share_1, len):
        """FSS Sign Evaluation when having len already known"""
        # Input parameters threshold, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len

        # Create parties
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

        # Send the shares to the parties
        # Calculate the performance and communication
        # t_sh_0 = time.perf_counter()
        P0.z_j = z_0
        P1.z_j = z_1

        # t_sh_1 = time.perf_counter()
        # t_share.append(t_sh_1 - t_sh_0)
        # t_sh_0 = time.perf_counter()

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        # Evaluation with FSS IC gate
        # and calculate the performance
        # t_cp_0 = time.perf_counter()

        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        # t_cp_1 = time.perf_counter()
        # t_comp.append(t_cp_1 - t_cp_0)

        # Construct the output of both parties
        o = P0.o_j + P1.o_j
        # t_sh_1 = time.perf_counter()
        # t_share.append(os.path.getsize("t_share_size.txt"))
        # t_share.append((t_sh_1 - t_sh_0))

        """ with open('t_DOP_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(P0.z_hat_j) + "\n")
            saveFile.write(str(P1.z_hat_j) + "\n")
            # saveFile.write("\n")

        t_share.append(os.path.getsize("t_DOP_size.txt"))"""

        return o

    def fss_evaluation_without_len(self, share_0, share_1):
        """FSS Sign Evaluation when not having len already known"""
        # Input parameters threshold, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len(z_0)

        # Create parties
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

        # Send the shares to the parties
        # and calculate the performance and communication
        # t_sh_0 = time.perf_counter()
        K = len(z_0)
        P0.z_j = z_0
        P1.z_j = z_1

        # t_sh_1 = time.perf_counter()
        # t_share.append(t_sh_1 - t_sh_0)
        # t_sh_0 = time.perf_counter()

        # Mask the public input to FSS gate
        P0.z_hat_j = P0.z_j + P0.r_in_j
        P1.z_hat_j = P1.z_j + P1.r_in_j

        P1.z_hat_nj = P0.z_hat_j
        P0.z_hat_nj = P1.z_hat_j

        # Evaluation with FSS IC gate
        # and calculate the performance
        # t_cp_0 = time.perf_counter()

        P1.o_j = funshade.eval_sign(K, P1.j, P1.k_j, P1.z_hat_j, P1.z_hat_nj)
        P0.o_j = funshade.eval_sign(K, P0.j, P0.k_j, P0.z_hat_j, P0.z_hat_nj)

        # t_cp_1 = time.perf_counter()
        # t_comp.append(t_cp_1 - t_cp_0)

        # t_sh_1 = time.perf_counter()
        # t_share.append(os.path.getsize("t_share_size.txt"))
        # t_share.append((t_sh_1 - t_sh_0))

        """ with open('t_DOP_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(P0.z_hat_j) + "\n")
            saveFile.write(str(P1.z_hat_j) + "\n")
            # saveFile.write("\n")

        t_share.append(os.path.getsize("t_DOP_size.txt"))"""

        return P0.o_j, P1.o_j

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError("multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))

        if self.data_min_cols < 100:
            raise ValueError("data_min_cols must be >= 100, got {}".format(self.data_min_cols))















