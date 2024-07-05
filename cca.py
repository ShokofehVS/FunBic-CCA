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
import time
import os

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
        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)
        # min_value = np.min(data)
        # max_value = np.max(data)
        # self._validate_parameters()

        # Create parties
        class party:
            def __init__(self, d: int):
                self.d = d

        P_0 = party(0)
        P_1 = party(1)

        # Generate secret shares of the input data
        rng = np.random.default_rng(seed=42)
        # in_0 = rng.integers(np.iinfo(funshade.DTYPE).min,
        #                    np.iinfo(funshade.DTYPE).max, size=(n_rows, n_cols), dtype=funshade.DTYPE)

        # in_0 = rng.integers(0, 800, size=(num_rows, num_cols))



        biclusters = []
        t_share, t_score, t_comp = [], [], []


        for i in range(self.num_biclusters):

            in_0 = rng.integers(np.iinfo(funshade.DTYPE).min,
                                np.iinfo(funshade.DTYPE).max, size=(num_rows, num_cols), dtype=funshade.DTYPE)
            in_1 = data - in_0

            num_row_0, num_col_0 = in_0.shape
            num_row_1, num_col_1 = in_1.shape

            P_0.in_d = in_0
            P_1.in_d = in_1

            rows_0 = np.ones(num_row_0, dtype=bool)
            cols_0 = np.ones(num_col_0, dtype=bool)

            rows_1 = np.ones(num_row_1, dtype=bool)
            cols_1 = np.ones(num_col_1, dtype=bool)

            self._multiple_node_deletion(in_0, in_1, rows_0, cols_0, rows_1, cols_1, self.msr_threshold)
            # self._node_addition(data, rows, cols)

            # row_0_indices = np.nonzero(rows_0)[0]
            # col_0_indices = np.nonzero(cols_0)[0]

            rows = rows_0
            cols = cols_0

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
            # print(max(self.sec_param))

        # Write the performance time
        with open('human_performance.txt', 'a') as saveFile:
            saveFile.write("\n")
            saveFile.write("Parties size:" + str(np.mean(t_share)) + "\n")
            # saveFile.write("t_score:" + str(np.mean(t_score)) + "\t")
            # saveFile.write("t_comp:" + str(np.mean(t_comp)) + "\t")
            # saveFile.write("\n")

        return Biclustering(biclusters)

    def _multiple_node_deletion(self, P_in_0, P_in_1, rows_0, cols_0, rows_1, cols_1, msr_thr):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described
        in the original paper)"""
        # Calculate MSR
        residue_0 = self._calculate_residue(P_in_0, rows_0, cols_0)
        residue_1 = self._calculate_residue(P_in_1, rows_1, cols_1)

        msr_0, row_msr_0, col_msr_0 = self._calculate_msr(residue_0 + residue_1)
        msr_1, row_msr_1, col_msr_1 = self._calculate_msr(residue_0 + residue_1)

        # Check whether the MSR is below or equal to threshold
        stop_itr_0 = msr_thr - msr_0
        stop_itr_1 = msr_thr - msr_1
        stop = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

        while not stop:
            cols0_st = np.copy(cols_0)
            rows0_st = np.copy(rows_0)

            cols1_st = np.copy(cols_1)
            rows1_st = np.copy(rows_1)

            # Check whether nodes are below 100
            if len(rows_0) or len(cols_0) >= self.data_min_cols:

                # Check whether rows are ready to remove
                r2remove_con_0 = row_msr_0 - self.multiple_node_deletion_threshold * msr_0
                r2remove_con_1 = row_msr_1 - self.multiple_node_deletion_threshold * msr_1
                fss_rs_rows = self.fss_evaluation_without_len(r2remove_con_0, r2remove_con_1)

                # Then remove those which are in the range of removal
                rows2remove_fss = np.nonzero(fss_rs_rows)
                row_indices = np.nonzero(rows_0)[0]
                rows2remove = row_indices[rows2remove_fss]
                rows_0[rows2remove] = False
                rows_1[rows2remove] = False

                # Recalculate the score
                residue_0 = self._calculate_residue(P_in_0, rows_0, cols_0)
                residue_1 = self._calculate_residue(P_in_1, rows_1, cols_1)

                msr_0, row_msr_0, col_msr_0 = self._calculate_msr(residue_0 + residue_1)
                msr_1, row_msr_1, col_msr_1 = self._calculate_msr(residue_0 + residue_1)

                # Check whether columns are ready to remove
                c2remove_con_0 = col_msr_0 - self.multiple_node_deletion_threshold * msr_0
                c2remove_con_1 = col_msr_1 - self.multiple_node_deletion_threshold * msr_1
                fss_rs_cols = self.fss_evaluation_without_len(c2remove_con_0, c2remove_con_1)

                # Then remove those which are in the range of removal
                cols2remove_fss = np.nonzero(fss_rs_cols)
                col_indices = np.nonzero(cols_0)[0]
                cols2remove = col_indices[cols2remove_fss]
                cols_0[cols2remove] = False
                cols_1[cols2remove] = False

            # Recalculate the score
            residue_0 = self._calculate_residue(P_in_0, rows_0, cols_0)
            residue_1 = self._calculate_residue(P_in_1, rows_1, cols_1)

            msr_0, row_msr_0, col_msr_0 = self._calculate_msr(residue_0 + residue_1)
            msr_1, row_msr_1, col_msr_1 = self._calculate_msr(residue_0 + residue_1)

            # Check whether the MSR is below or equal to threshold or any nodes have been removed
            stop_itr_0 = msr_thr - msr_0
            stop_itr_1 = msr_thr - msr_1
            stop_con1 = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

            stop_itr_r0 = np.sum(rows_0 - (rows0_st).astype(int))
            stop_itr_r1 = np.sum(rows_1 - (rows1_st).astype(int))
            stop_it_r = self.fss_evaluation(stop_itr_r0, stop_itr_r1, 1)

            stop_itr_c0 = np.sum(cols_0 - (cols0_st).astype(int))
            stop_itr_c1 = np.sum(cols_1 - (cols1_st).astype(int))
            stop_it_c = self.fss_evaluation(stop_itr_c0, stop_itr_c1, 1)

            stop_it_nodes = stop_it_r * stop_it_c
            stop_it = (stop_con1 * stop_it_nodes) + (stop_con1 + stop_it_nodes)
            stop = np.divmod(stop_it, 2)[1]

    def _node_addition(self, data, rows, cols):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
        the original paper)"""
        stop = False
        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            msr, _, _ = self._calculate_msr(data, rows, cols)
            # col_msr = self._calculate_msr_col_addition(data, rows, cols, t_score)
            #
            # cols2add_fss = self.fss_evaluation(msr - col_msr, t_share, t_comp)
            # cols2add = np.nonzero(cols2add_fss)
            # cols[cols2add] = True
            #
            # msr, _, _ = self._calculate_msr(data, rows, cols, t_score)
            # row_msr = self._calculate_msr_row_addition(data, rows, cols, t_score)
            #
            # row2add_fss = self.fss_evaluation(msr - row_msr, t_share, t_comp)
            # rows2add = np.nonzero(row2add_fss)
            # rows[rows2add] = True

            if np.all(rows == rows_old) and np.all(cols == cols_old):
                stop = True

    def _calculate_residue(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""
        # Calculate performance
        # t_sc_0 = time.perf_counter()

        sub_data = data[rows][:, cols]
        l_r = len(rows)
        l_c = len(cols)
        n_elements = l_r * l_c

        """msr = np.sum((sub_data * n_elements - sub_data.sum(axis=0) * l_c
                          - np.expand_dims(sub_data.sum(axis=1), axis=1) * l_r + sub_data.sum()) ** 2) / (
                              n_elements ** 3)

        row_msr = np.sum((sub_data * n_elements - sub_data.sum(axis=0) * l_c
                              - np.expand_dims(sub_data.sum(axis=1), axis=1) * l_r + sub_data.sum()) ** 2,
                             axis=1) / ((n_elements ** 2) * l_c)

        col_msr = np.sum((sub_data * n_elements - sub_data.sum(axis=0) * l_c
                              - np.expand_dims(sub_data.sum(axis=1), axis=1) * l_r + sub_data.sum()) ** 2, axis=0) / (
                                  (n_elements ** 2) * l_r)"""

        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data, axis=0)

        residues = (sub_data - row_means[:, np.newaxis] - col_means + data_mean)

        rng = np.random.default_rng(seed=42)
        mask = rng.integers(0, 1, size=residues.shape)

        return mask + residues

    def _calculate_msr(self, residues):

        squared_residues = residues * residues

        msr = np.mean(squared_residues)
        row_msr = np.mean(squared_residues, axis=1)
        col_msr = np.mean(squared_residues, axis=0)

        """ t_sc_1 = time.perf_counter()
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
        return msr, row_msr, col_msr

    def _calculate_msr_col_addition(self, data, rows, cols, t_score):
        """Calculate the mean squared residues of the columns for the node addition step."""
        # Calculate performance
        t_sc_0 = time.perf_counter()
        sub_data = data[rows][:, cols]
        sub_data_rows = data[rows]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data_rows, axis=0)

        col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean
        col_squared_residues = col_residues * col_residues
        col_msr = np.mean(col_squared_residues, axis=0)

        t_sc_1 = time.perf_counter()
        # t_score.append(t_sc_1 - t_sc_0)

        with open('t_score_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(col_msr) + "\n")
            # saveFile.write("\n")

            # t_sh_1 = time.perf_counter()
            # t_share.append(t_sh_1 - t_sh_0)
        t_score.append(os.path.getsize("t_score_size.txt"))

        return col_msr

    def _calculate_msr_row_addition(self, data, rows, cols, t_score):
        """Calculate the mean squared residues of the rows and of the inverse of the rows for
        the node addition step."""
        # Calculate performance
        t_sc_0 = time.perf_counter()
        sub_data = data[rows][:, cols]
        sub_data_cols = data[:, cols]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data_cols, axis=1)
        col_means = np.mean(sub_data, axis=0)

        row_residues = sub_data_cols - row_means[:, np.newaxis] - col_means + data_mean
        row_squared_residues = row_residues * row_residues
        row_msr = np.mean(row_squared_residues, axis=1)

        t_sc_1 = time.perf_counter()
        # t_score.append(t_sc_1 - t_sc_0)

        with open('t_score_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(row_msr) + "\n")
            # saveFile.write("\n")

            # t_sh_1 = time.perf_counter()
            # t_share.append(t_sh_1 - t_sh_0)
        t_score.append(os.path.getsize("t_score_size.txt"))

        return row_msr

    def del_add_node(self, P_0, P_1, u, v, K, l, gamma):
        d_u0, d_u1, d_v0, d_v1, d_w0, d_w1, r_in0, r_in1, k0, k1 = funshade.setup(K, l, gamma)

        # Distribute randomness to (P0, P1)
        P_0.d_u_j  = d_u0;            P_1.d_u_j  = d_u1
        P_0.d_v_j  = d_v0;            P_1.d_v_j  = d_v1
        P_0.d_w_j  = d_w0;             P_1.d_w_j = d_w1
        P_0.r_in_j = r_in0;           P_1.r_in_j = r_in1
        P_0.k_j    = k0;              P_1.k_j    = k1
        P_0.d_u    = d_u0 + d_u1;     P_1.d_v    = d_v0 + d_v1

        # (2) Get and secret share the Delta share
        P_0.u       = u
        P_0.D_u = funshade.share(K, l, P_0.u, P_0.d_u)
        P_1.D_u = P_0.D_u

        P_1.v       = v
        P_1.D_v = funshade.share(K, l, P_1.v, P_1.d_v)
        P_0.D_v = P_1.D_v

        rem_add_1 = funshade.eval_dist(K, l, P_1.j,
                           P_1.r_in_j, P_1.D_u, P_1.D_v, P_1.d_u_j, P_1.d_v_j, P_1.d_w_j)

        rem_add_0 = funshade.eval_dist(K, l, P_0.j,
                           P_0.r_in_j, P_0.D_u, P_0.D_v, P_0.d_u_j, P_0.d_v_j, P_0.d_w_j)

        return rem_add_0, rem_add_1

    def fss_evaluation(self, share_0, share_1, len):
        """Calculate the function secrete sharing of the secret share of input and output whether to remove/ add nodes
        from/ to matrix"""
        # Input parameters threshold, inputs, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len

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

        # # Secrete share the input
        # rng = np.random.default_rng(seed=42)
        # z_0 = rng.integers(np.iinfo(funshade.DTYPE).min,
        #                    np.iinfo(funshade.DTYPE).max, size=K, dtype=funshade.DTYPE)
        # z_1 = z - z_0

        # Send the shares to the parties
        # and calculate the performance and communication
        # t_sh_0 = time.perf_counter()
        P0.z_j = z_0
        P1.z_j = z_1

        # t_sh_1 = time.perf_counter()
        # t_share.append(t_sh_1 - t_sh_0)

        # Mask the public input to FSS gate
        # t_sh_0 = time.perf_counter()

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
        t_sh_1 = time.perf_counter()
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
        """Calculate the function secrete sharing of the secret share of input and output whether to remove/ add nodes
        from/ to matrix"""
        # Input parameters threshold, inputs, and length of matrix
        gamma = 0
        z_0 = share_0.astype(funshade.DTYPE)
        z_1 = share_1.astype(funshade.DTYPE)
        K = len(z_0)
        # K = len

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

        # # Secrete share the input
        # rng = np.random.default_rng(seed=42)
        # z_0 = rng.integers(np.iinfo(funshade.DTYPE).min,
        #                    np.iinfo(funshade.DTYPE).max, size=K, dtype=funshade.DTYPE)
        # z_1 = z - z_0

        # Send the shares to the parties
        # and calculate the performance and communication
        # t_sh_0 = time.perf_counter()
        K = len(z_0)
        P0.z_j = z_0
        P1.z_j = z_1

        # t_sh_1 = time.perf_counter()
        # t_share.append(t_sh_1 - t_sh_0)

        # Mask the public input to FSS gate
        # t_sh_0 = time.perf_counter()

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
        t_sh_1 = time.perf_counter()
        # t_share.append(os.path.getsize("t_share_size.txt"))
        # t_share.append((t_sh_1 - t_sh_0))

        """ with open('t_DOP_size.txt', 'w') as saveFile:
            # saveFile.write("\n")
            saveFile.write(str(P0.z_hat_j) + "\n")
            saveFile.write(str(P1.z_hat_j) + "\n")
            # saveFile.write("\n")

        t_share.append(os.path.getsize("t_DOP_size.txt"))"""

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


