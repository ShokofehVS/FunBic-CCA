"""
    FunBic-CCA: A Python library of privacy-preserving biclustering algorithm (Cheng and Church) with Function Secret
    Sharing

    Copyright (C) 2024  Shokofeh VahidianSadegh

    This file is part of FunBic-CCA.

"""
from _base import BaseBiclusteringAlgorithm
from models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
import sycret
import numpy as np
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

        # Helper vectors
        biclusters = []
        t_shareMSR, t_shareEval, t_size = [], [], []

        # For number of biclusters do:
        for i in range(self.num_biclusters):
            # Shape of data and min/ max of data
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

            # Performance analysis
            with open('result_size.txt', 'w') as saveFile:
                saveFile.write(str(P_0.aij_0) + "\n")
                saveFile.write(str(P_1.aij_1) + "\n")
            t_size.append(os.path.getsize("result_size.txt"))

            # Steps including single, multiple deletion/ addition
            P_0.bij_0, P_1.bij_1, len_row  = (
                self._multiple_node_deletion(P_0.aij_0, P_1.aij_1, rows_0, cols_0,  rows_1, cols_1, self.msr_threshold))

            P_0.cij_0, P_1.cij_1, len_row, len_col  = (
                self._single_node_deletion(P_0.bij_0, P_1.bij_1, len_row, self.msr_threshold))

            P_0.dij_0, P_1.dij_1  = (
                self._node_addition(P_0.cij_0, P_1.cij_1, in_0, in_1, len_row, len_col))

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

        # Write about communication size/ rounds
        with open('yeast_performance.txt', 'a') as saveFile:
            saveFile.write("\n")
            saveFile.write("Number of bits     :" + str(self.highest_range) + "\n")
            saveFile.write("Communication size :" + str(np.mean(t_size)) + "\n")


        return Biclustering(biclusters)


    def _single_node_deletion(self, in_0, in_1, len_row, msr_thr):
        """Performs the single row/column deletion step (this is a direct implementation of the Algorithm 1 described
           in the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape
        num_row_1, num_col_1 = in_1.shape

        # Calculate the scores by having inputs including secret shares of matrix, and length of rows, columns
        msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1 = (self._calculate_scores_del(in_0, in_1, len_row, num_col_0))

        # STOP function -- Check whether the MSR is below or equal to threshold
        stop_itr_0 = msr_thr - msr_0
        stop_itr_1 = msr_thr - msr_1
        stop = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

        if stop:
            # No node has been removed
            return in_0, in_1, len_row, num_col_0

        else:
            while not stop:
                # Find the argmax of nodes whichever having the largest scores
                row_max_msr = self._amx(row_msr_0, row_msr_1)
                col_max_msr = self._amx(col_msr_0, col_msr_1)

                # Check score of row/ column with maximum values to remove that particular node
                eval_node_0 = row_msr_0[row_max_msr] - col_msr_0[col_max_msr]
                eval_node_1 = row_msr_1[row_max_msr] - col_msr_1[col_max_msr]
                sdel0, sdel1 = self.fss_evaluation_sdel(eval_node_0, eval_node_1, 1)

                # Check whether row_msr[row_max_msr] >= col_msr[col_max_msr] or not
                cond = self._equality_check_2(sdel0, sdel1, 0, 0, 1)

                # Remove the row/ column based on the result of evaluation 0 => remove row, 1 => remove column
                r2del, c2del = [], []
                if cond == 0:
                    # Because some rows might be zero now, let's ignore them in single node deletion
                    for idxr in range(num_row_0):
                        srdel = self._equality_check_2(in_0[idxr], in_1[idxr], 0, 0, num_col_0)
                        if srdel.all() == 1:
                            pass
                        else:
                            r2del.append(idxr)
                    r2del_ind = r2del[row_max_msr]
                    in_0[r2del_ind] = 0;                                in_1[r2del_ind] = 0
                    len_row        -= 1;                                   len_row     -= 1

                else:
                    # Transpose secret shared input matrices before removing column
                    transposed_in_0 = in_0.T
                    transposed_in_1 = in_1.T

                    # Because some columns might be zero now, let's ignore them in single node deletion
                    for idxc in range(num_col_0):
                        scdel =  self._equality_check_2(transposed_in_0[idxc], transposed_in_1[idxc],
                                                        0, 0, num_row_0)
                        if scdel.all() == 1:
                            pass
                        else:
                            c2del.append(idxc)
                    c2del_ind = c2del[col_max_msr]
                    transposed_in_0[c2del_ind] = 0;                       transposed_in_1[c2del_ind] = 0

                    # Return the transposed matrices to normal
                    in_0 = transposed_in_0.T
                    in_1 = transposed_in_1.T
                    num_col_0    -= 1;                                    num_col_1    -= 1

                # Recalculate the scores by having inputs including secret shares of matrix, and length of rows, columns
                msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1 = \
                    (self._calculate_scores_del(in_0, in_1, len_row, num_col_0))

                # Recheck the stop function
                stop_itr_0 = msr_thr - msr_0
                stop_itr_1 = msr_thr - msr_1
                stop = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

        return in_0, in_1, len_row, num_col_0


    def _multiple_node_deletion(self, in_0, in_1, rows_0, cols_0, rows_1, cols_1, msr_thr):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described
           in the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape
        num_row_1, num_col_1 = in_1.shape

        # MSRs computation when NO nodes are removed (having exact rows/ columns length)
        msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1  = self._scores_before_steps(in_0, in_1)

        # STOP function -- Check whether the MSR is below or equal to threshold
        stop_itr_0 = msr_thr - msr_0
        stop_itr_1 = msr_thr - msr_1
        stop       = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

        if stop:
            # No nodes have been removed so return length of rows without change
            return in_0, in_1, num_row_0

        else:
            while not stop:
                # Store previous values of matrices for equality check
                cp_in_0 = np.copy(in_0);                 cp_in_1 = np.copy(in_1)

                # FSS IC gate to check which rows should be removed
                r2remove_con_0 = self.multiple_node_deletion_threshold * msr_0 - row_msr_0
                r2remove_con_1 = self.multiple_node_deletion_threshold * msr_1 - row_msr_1
                fss_rs_rows_0, fss_rs_rows_1 = self.fss_evaluation_without_len(r2remove_con_0, r2remove_con_1)

                # Remove the rows based on the result of evaluation 1 => remove row, 0 => nothing
                nr2del = self._equality_check_2(fss_rs_rows_0, fss_rs_rows_1, 0, 0, num_row_0)

                # Because some rows might be zero now, let's ignore them in multiple node deletion
                # First iteration; finds those zeros rows
                # Second iteration; try to mask those rows that are not removed with zeros
                r2del = []
                for idxr in range(num_row_0):
                    srdel = self._equality_check_2(in_0[idxr], in_1[idxr], 0, 0, num_col_0)
                    if srdel.all() == 1:
                        pass
                    else:
                        r2del.append(idxr)

                total_len_row = len(r2del)
                for idxr in range(len(r2del)):
                    i = nr2del[idxr]
                    if i == 1:
                        r2del_ind = r2del[idxr]
                        in_0[r2del_ind] = 0;               in_1[r2del_ind] = 0
                        total_len_row  -= 1
                    else:
                        pass


                # Check whether columns are above 100 then apply node deletion on them
                if len(cols_0) >= self.data_min_cols:
                    in_0, in_1 = self._cols2Remove(in_0, in_1, rows_0, rows_1, cols_0, cols_1,
                                                   num_row_0, num_col_0, num_row_1, num_col_1)

                # Recalculate the scores (the columns by default have not been removed) to be used in stop function below
                msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1 =  (self._calculate_scores_del
                                                                             (in_0, in_1,
                                                                              total_len_row,
                                                                              num_col_0))


                # First stop function; check whether any nodes have been removed (equality of current, previous nodes)
                stop_con1 = self._equality_check(in_0, in_1, cp_in_0, cp_in_1)

                # Second stop function; check also the MSR is below/equal to threshold
                stop_itr_0 = msr_thr - msr_0
                stop_itr_1 = msr_thr - msr_1
                stop_con2 = self.fss_evaluation(stop_itr_0, stop_itr_1, 1)

                # OR between the above-calculated stop functions
                stop = stop_con1 or stop_con2


        return in_0, in_1, total_len_row


    def _node_addition(self, bij_0, bij_1, in_0, in_1, total_len_row, len_col):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
           the original paper)"""
        # Secret shared inputs' shapes
        num_row_0, num_col_0 = in_0.shape
        num_row_1, num_col_1 = in_1.shape

        # Calculate score for whole matrix and that of columns
        msr_0, msr_1, _, _, col_msr_0, col_msr_1 = (self._calculate_scores_del(bij_0, bij_1, total_len_row, len_col))

        # FSS IC gate to check which columns should be added
        r2add_con_0 = msr_0 - col_msr_0
        r2add_con_1 = msr_1 - col_msr_1
        fss_rs_col_add_0, fss_rs_col_add_1 = self.fss_evaluation_without_len(r2add_con_0, r2add_con_1)

        # Then follow similar steps in deletion; except return those have been removed
        # Add the columns based on the result of evaluation 1 => nothing, 0 => add columns
        nc2add = self._equality_check_2(fss_rs_col_add_0, fss_rs_col_add_1, 0, 0, total_len_row)

        # Transpose secret shared input matrices before adding column
        transposed_in_0 = bij_0.T
        transposed_in_1 = bij_1.T

        # Because some columns might be zero now, let's ignore them in addition
        # First iteration; finds those zeros columns
        # Second iteration; try to add them with original matrix values
        c2add = []
        for idxc in range(len_col):
            scdel = self._equality_check_2(transposed_in_0[idxc], transposed_in_1[idxc],0, 0, total_len_row)
            if scdel.all() == 1:
                pass
            else:
                c2del.append(idxc)
        c2del_ind = c2del[col_max_msr]
        transposed_in_0[c2del_ind] = 0;
        transposed_in_1[c2del_ind] = 0

        # Return the transposed matrices to normal
        in_0 = transposed_in_0.T
        in_1 = transposed_in_1.T
        num_col_0 -= 1;
        num_col_1 -= 1



        # Calculate score for whole matrix and that of rows
        msr_0, msr_1, row_msr_0, row_msr_1, _, _ =  (self._calculate_scores_del(bij_0, bij_1, total_len_row, len_col))

        # FSS IC gate to check which rows should be added
        r2add_con_0 = msr_0 - row_msr_0
        r2add_con_1 = msr_1 - row_msr_1
        fss_rs_rows_add_0, fss_rs_rows_add_1 = self.fss_evaluation_without_len(r2add_con_0, r2add_con_1)

        # Then follow similar steps in deletion; except return those have been removed
        # Add the rows based on the result of evaluation 1 => nothing, 0 => add rows
        nr2add = self._equality_check_2(fss_rs_rows_add_0, fss_rs_rows_add_1, 0, 0, num_row_0)

        # Because some rows might be zero now, let's ignore them in addition
        # First iteration; finds those zeros rows
        # Second iteration; try to add them with original matrix values
        r2add = []
        for idxr in range(total_len_row):
            sradd = self._equality_check_2(bij_0[idxr], bij_1[idxr], 0, 0, num_col_0)
            if r2add.all() == 1:
                r2add.append(idxr)
            else:
                pass

        for idxr in range(len(r2add)):
            i = nr2add[idxr]
            if i == 1:
                r2add_ind = nr2add[idxr]
                bij_0[r2add_ind] = in_0[r2add_ind]
                bij_1[r2add_ind] = in_1[r2add_ind]
                total_len_row += 1
            else:
                pass


        return bij_0, bij_1

    def _amx(self, in_0, in_1):
        """Calculate Argmax of scores of the rows, of the columns."""
        # Initial values
        m = len(in_0)
        argmx_con_0, argmx_con_1, delta_j  = [], [], []

        # Argmax according to AriaNN algorithm 6
        for j in range(m):
            for i in range(m):
                if i != j:
                    argmx_con_0.append(in_0[j] - in_0[i])
                    argmx_con_1.append(in_1[j] - in_1[i])
                else:
                    pass
            node_max_0, node_max_1 = self.fss_evaluation_without_len(np.array(argmx_con_0), np.array(argmx_con_1))
            s_j_0 = np.sum(node_max_0);                                 s_j_1 = np.sum(node_max_1)
            delta_j.append(self._equality_check_2(s_j_0, s_j_1, m-1, 0, 1))
            argmx_con_0, argmx_con_1 = [], []
            if delta_j[j] == 1:
                arg_max_res = j
                return arg_max_res


    def _scores_before_steps(self, in_0, in_1):
        """Calculate scores of the rows, of the columns and of the full data matrix before any steps"""
        # Note that all means are converted to integers to avoid overflow
        # Mean values
        data_mean_0 = np.mean(in_0).astype(int);                        data_mean_1 = np.mean(in_1).astype(int)
        row_means_0 = np.mean(in_0, axis=1).astype(int);                row_means_1 = np.mean(in_1, axis=1).astype(int)
        col_means_0 = np.mean(in_0, axis=0).astype(int);                col_means_1 = np.mean(in_1, axis=0).astype(int)

        # Residues
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

    def _calculate_scores_del(self, in_0, in_1, len_row, num_col_0):
        """Calculate scores of the rows, of the columns and of the full data matrix after rows deletion"""
        # ** Mean values calculation starts by knowing the length of columns but not rows
        # Get the sum of whole data matrix
        whole_sum_0 = np.sum(in_0)
        whole_sum_1 = np.sum(in_1)

        # Mean of whole matrix by dividing the sum to length of matrix
        data_mean_0 = (whole_sum_0 / (len_row * num_col_0)).astype(int)
        data_mean_1 = (whole_sum_1 / (len_row * num_col_0)).astype(int)

        # Mean of rows by knowing the length of columns
        row_mean_0 = (np.sum(in_0, axis=1) / num_col_0).astype(int)
        row_mean_1 = (np.sum(in_1, axis=1) / num_col_0).astype(int)

        # Mean of columns by dividing the sum to length of rows
        col_mean_0 = (np.sum(in_0, axis=0) / (len_row)).astype(int)
        col_mean_1 = (np.sum(in_1, axis=0) / (len_row)).astype(int)

        # ** Residue calculation starts by having all mean values
        residue_0 = in_0 - row_mean_0[:, np.newaxis] - col_mean_0 + data_mean_0
        residue_1 = in_1 - row_mean_1[:, np.newaxis] - col_mean_1 + data_mean_1

        # ** Continue doing squaring by power of 2 residue and joint multiplication
        squared_residue_0 = np.copy(residue_0)
        squared_residue_1 = np.copy(residue_1)

        for idxr in range(residue_0.shape[0]):
            squared_residue_0[idxr], squared_residue_1[idxr] = self.secSquare_vector(residue_0[idxr],
                                                                                     residue_1[idxr])

        # ** Another mean values calculation, this time for squared residues
        msr_0 = (np.mean(squared_residue_0)).astype(int)
        msr_1 = (np.mean(squared_residue_1)).astype(int)

        row_msr_0 = (np.mean(squared_residue_0, axis=1)).astype(int)
        row_msr_1 = (np.mean(squared_residue_1, axis=1)).astype(int)

        col_msr_0 = (np.mean(squared_residue_0, axis=0)).astype(int)
        col_msr_1 = (np.mean(squared_residue_1, axis=0)).astype(int)


        return msr_0, msr_1, row_msr_0, row_msr_1, col_msr_0, col_msr_1


    def _equality_check(self, in_0, in_1, cp_in_0, cp_in_1):
        """Determine equality of matrix before and after node deletion; usage in stop function of multiple deletion"""
        # Determine the number of secret shared elements for keys
        n_row, n_cols = in_0.shape
        n_element     = n_row * n_cols

        # An instance of DPF gate for equality check with 6 threads
        eq = sycret.EqFactory(n_threads=6)

        # Generation of DPF keys
        keys_a, keys_b = eq.keygen(n_element)

        # Alpha based on generated keys
        alpha = eq.alpha(keys_a, keys_b)

        # Secret share the Alpha
        rng = np.random.default_rng(seed=42)
        e_rin_0 = rng.integers(1, self.highest_range, size=n_element, dtype="int64")
        e_rin_1 = alpha - e_rin_0

        # Input shares for DPF gate
        dpf_in0 = in_0 - cp_in_0
        dpf_in1 = in_1 - cp_in_1

        # Convert to flatten vectors
        dpf_in0 = dpf_in0.flatten()
        dpf_in1 = dpf_in1.flatten()

        # Add the mask to secret shares before reconstruction
        mdpf_in0 = dpf_in0 + e_rin_0
        mdpf_in1 = dpf_in1 + e_rin_1

        # Now exchange the masked input to DPF FSS gate
        f_out = mdpf_in0 + mdpf_in1

        # Apply DPF for equality check
        r_a, r_b = (
            eq.eval(0, f_out, keys_a),
            eq.eval(1, f_out, keys_b),
        )
        r_eq = (r_a + r_b) % (2 ** (eq.N * 8))

        # Check whether all nodes are the same or there are any changes in the matrices
        if np.sum(r_eq) == n_element:
            stop = True
        else:
            stop = False


        return stop

    def _equality_check_2(self, in_0, in_1, cp_in_0, cp_in_1, n_element):
        """Determine equality of vectors; usage in deletion steps"""
        # An instance of DPF gate for equality check with 6 threads
        eq = sycret.EqFactory(n_threads=6)

        # Generation of DPF keys
        keys_a, keys_b = eq.keygen(n_element)

        # Alpha based on generated keys
        alpha = eq.alpha(keys_a, keys_b)

        # Secret share the Alpha
        rng = np.random.default_rng(seed=42)
        e_rin_0 = rng.integers(1, self.highest_range, size=n_element, dtype="int64")
        e_rin_1 = alpha - e_rin_0

        # Input shares for DPF gate
        dpf_in0 = in_0 - cp_in_0
        dpf_in1 = in_1 - cp_in_1

        # Add the mask to secret shares before reconstruction
        mdpf_in0 = dpf_in0 + e_rin_0
        mdpf_in1 = dpf_in1 + e_rin_1

        # Now exchange the masked input to DPF FSS gate
        f_out = mdpf_in0 + mdpf_in1

        # Apply DPF for equality check
        r_a, r_b = (
            eq.eval(0, f_out, keys_a),
            eq.eval(1, f_out, keys_b),
        )
        r_eq = (r_a + r_b) % (2 ** (eq.N * 8))


        return r_eq


    def _cols2Remove(self, in_0, in_1, rows_0, rows_1, cols_0, cols_1, num_row_0, num_col_0, num_row_1, num_col_1):
        """Calculate multiple node deletion for columns if their length is above 100"""
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


    def _calculate_residue(self, data, rows, cols, multidel, coladd, rowadd):
        """MUST BE REPLACED WITH NEW IMPLEMENTATION --- ONLY USE CASE IN ALGORITHM STEP FOR COLUMNS"""
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
        """MUST BE REPLACED WITH NEW IMPLEMENTATION  --- ONLY USE CASE IN ALGORITHM STEP FOR COLUMNS"""
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
        """Generation of Beaver Triples and secured multiplication for doing squaring."""
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
        """FSS IC Sign Evaluation when having known length of input vector"""
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

    def fss_evaluation_without_len(self, share_0, share_1):
        """FSS Sign Evaluation without having length of input vector."""
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
        K = len(z_0)
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


        return P0.o_j, P1.o_j

    def fss_evaluation_sdel(self, share_0, share_1, len):
        """FSS IC Sign Evaluation when having known length of input vector particularly for single node deletion"""
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















