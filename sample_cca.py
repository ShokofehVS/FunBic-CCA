import numpy as np
import seaborn as sn
from cca import ChengChurchAlgorithm, ModifiedChengChurchAlgorithm
# import load_yeast_tavazoie, load_prelic

# load yeast data used in the original Cheng and Church's paper
# data = load_yeast_tavazoie().values
n_rows, n_cols = 20, 17
n_elements = n_rows*n_cols
# np.random.seed(42) # Fixed seed for reproducibility
# data = np.random.randint(0, 1, size=(n_rows, n_cols))
data = np.random.randint(0, 100, size=(n_rows, n_cols))
# print(data)

# print(data)
# missing value imputation suggested by Cheng and Church
"""missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=1, size=len(missing[0]))
"""
# creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
cca = ModifiedChengChurchAlgorithm(num_biclusters=10, msr_threshold=0.1, multiple_node_deletion_threshold=1.2,
                                   data_min_cols=100, alpha=0.05)
biclustering = cca.run(data)

print(biclustering)
