import numpy as np
# from _io import _biclustering_to_dict, _dict_to_biclustering
from models import Bicluster, Biclustering

# Data generation
n_rows, n_cols = 10, 17
n_elements = n_rows * n_cols
np.random.seed(42)  # Fixed seed for reproducibility
data = np.random.randint(0, 600, size=(n_rows, n_cols))

# Parameter selection and initialization
alpha = 1.2
delta = 300
num_biclusters = 100
biclusters = []

# Max and min for random distribution
min_value = np.min(data)
max_value = np.max(data)


# def write_results_in_file(self, biclustering):
#
#     # Save biclustering results to a file
#     biclustering_dict = _biclustering_to_dict(biclustering)
#     biclustering_list = biclustering_dict["biclusters"]
#     with open('CCA_result.out', 'w') as saveFile:
#         for bicluster in biclustering_list:
#             saveFile.write(str(bicluster))
#             saveFile.write("\n")


def mean_squared_residue(data, row, col):

    sub_data = data[row][:, col]
    row_mean = np.mean(sub_data, axis=1)
    col_mean = np.mean(sub_data, axis=0)
    data_mean = np.mean(sub_data)

    residue = sub_data - row_mean[:, np.newaxis] - col_mean + data_mean
    squared_residue = residue * residue

    msr = np.mean(squared_residue)
    msr_row = np.mean(squared_residue, axis=1)
    msr_col = np.mean(squared_residue, axis=0)

    return msr, msr_row, msr_col


def col_mean_squared_residue_add(data, row, col):

    sub_data = data[row][:, col]
    sub_data_row = data[row]

    row_mean = np.mean(sub_data_row, axis=1)
    col_mean = np.mean(sub_data, axis=0)
    data_mean = np.mean(sub_data)

    residue = sub_data - row_mean[:, np.newaxis] - col_mean + data_mean
    squared_residue = residue * residue

    msr = np.mean(squared_residue)
    msr_col = np.mean(squared_residue, axis=0)

    return msr_col


def row_mean_squared_residue_add(data, row, col):

    sub_data = data[row][:, col]
    sub_data_col = data[:, col]

    row_mean = np.mean(sub_data_col, axis=1)
    col_mean = np.mean(sub_data, axis=0)
    data_mean = np.mean(sub_data)

    residue = sub_data - row_mean[:, np.newaxis] - col_mean + data_mean
    squared_residue = residue * residue

    msr = np.mean(squared_residue)
    msr_row = np.mean(squared_residue, axis=1)
    #
    # residue_inv = - sub_data + row_mean - col_mean + data_mean
    # squared_residue_inv = np.square(residue_inv)

    # msr_inv = np.mean(squared_residue_inv)
    # msr_row_inv = np.mean(msr_inv, axis=1)

    return msr_row


def multi_node_del(data, row, col):
    row_copy = np.copy(row)
    col_copy = np.copy(col)

    row_index = np.nonzero(row)[0]
    col_index = np.nonzero(col)[0]

    msr, msr_r, msr_c = mean_squared_residue(data, row, col)

    stop = False

    if msr <= delta:
        stop = True

    while not stop:
        if len(row) or len(col) >= 100:
            row2remove = row_index[np.where(msr_r > (alpha * msr))]
            row[row2remove] = False

            msr, msr_r, msr_c = mean_squared_residue(data, row, col)

            col2remove = col_index[np.where(msr_c > (alpha * msr))]
            col[col2remove] = False

        if (msr <= delta) or (np.all(row_copy == row) and np.all(col_copy == col)):
            stop = True


# def single_node_del(self, data, row, col):
#     msr_r, msr_c, msr = self.mean_squared_residue(data, row, col)
#
#     col_index = np.nonzero(col)[0]
#     row_index = np.nonzero(row)[0]
#
#     stop = False
#
#     if msr <= self.delta:
#         stop = True
#
#     while not stop:
#         max_msr_r = np.argmax(msr_r)
#         max_msr_c = np.argmax(msr_c)
#
#         row2remove = row_index[max_msr_r]
#         row[row2remove] = False
#
#         col2remove = col_index[max_msr_c]
#         col[col2remove] = False
#
#         msr, msr_r, msr_c = self.mean_squared_residue(data, row, col)
#
#         if msr <= self.delta:
#             stop = True


def node_add(data, row, col):
    row_copy = np.copy(row)
    col_copy = np.copy(col)

    stop = False

    while not stop:
        msr, _, _ = mean_squared_residue(data, row, col)
        col_index = np.nonzero(col)[0]
        msr_c = col_mean_squared_residue_add(data, row, col)
        col2add = col_index[np.where(msr_c <= msr)]
        col[col2add] = True

        msr, msr_r, msr_c = mean_squared_residue(data, row, col)
        msr_r = row_mean_squared_residue_add(data, row, col)
        row_index = np.nonzero(row)[0]
        row2add = row_index[np.where(msr_r <= msr)]
        row[row2add] = True

        if (np.all(row_copy == row) and np.all(col_copy == col)):
            stop = True


for i in range(num_biclusters):

    row = np.ones(n_rows, dtype=bool)
    col = np.ones(n_cols, dtype=bool)

    multi_node_del(data, row, col) #WE HAVE A LOGICAL PROBLEM
    node_add(data, row, col)

print("CCA executed correctly")

#     row_indices = np.nonzero(row)[0]
#     col_indices = np.nonzero(col)[0]
#
#     if i < num_biclusters - 1:
#         bicluster_shape = (len(row_indices), len(col_indices))
#         data[row_indices[:, np.newaxis], col_indices] = np.random.uniform(low=min_value, high=max_value,
#                                                                           size=bicluster_shape)
#
# biclusters.append(Bicluster(row_indices, col_indices))

