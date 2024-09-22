import numpy as np
import cca
import origCCA
import pandas as pd
import accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import time
import funshade
import random
import warnings

from os.path import dirname, join
from urllib.request import urlopen

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Yeast Cell Cycle Data Set Generation
def load_yeast_tavazoie():
    """Load and return the yeast dataset (Tavazoie et al., 2000) used in the original biclustering study
    of Cheng and Church (2000) as a pandas.DataFrame. All elements equal to -1 are missing values. This
    dataset is freely available in http://arep.med.harvard.edu/biclustering/.

    Reference
    ---------
    Cheng, Y., & Church, G. M. (2000). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    Tavazoie, S., Hughes, J. D., Campbell, M. J., Cho, R. J., & Church, G. M. (1999). Systematic determination of genetic
    network architecture. Nature genetics, 22(3), 281-285.
    """
    module_dir = dirname(__file__)
    data = np.loadtxt(join(module_dir, 'yeast_tavazoie.txt'), dtype=np.double)
    genes = np.loadtxt(join(module_dir,'genes_yeast_tavazoie.txt'), dtype=str)
    return pd.DataFrame(data, index=genes)


# Human Gene Expression Data Set Generation
def load_human_data():
    with open("lymphoma.matrix.txt") as f:
        lines = f.read().strip().split('\n')
        lines = list(' -'.join(line.split('-')).split(' ') for line in lines)

        lines = list(list(int(i) for i in line if i) for line in lines)
        data = np.array(lines)
        if lines:
            if "true" in lines:
                print("There is a profane word in the document")

        return data


# Load data sets
data   = load_yeast_tavazoie().values
# data = load_human_data()

# Number of Elements in Matrix
row, col   = data.shape
n_elements = row * col

# missing value imputation suggested by Cheng and Church for Yeast data set
missing       = np.where((data <= 0.0))
data[missing] = np.random.randint(low=1, high=800, size=len(missing[0]))

# missing value imputation suggested by Cheng and Church for Human data set
"""generator = np.random.RandomState(0)
idx = np.where(data == 999)
data[idx] = generator.randint(1, 800, len(idx[0]))

missing = np.where((data <= 0.0))
data[missing] = np.random.randint(low=1, high=800, size=len(missing[0]))"""

# Avoid overflow
max_bit = 2 ** 3 # yeast data
# max_bit = 2 ** 3 # human data

# Run the secured and non-secure algorithms
data_transformed = np.random.randint(1, max_bit, size=(row, col), dtype="int64")
n_rows, n_cols   = data_transformed.shape

# Creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
cca = cca.ChengChurchAlgorithm(max_bit, num_biclusters=100, msr_threshold=300,
                               multiple_node_deletion_threshold=1.2, data_min_cols=100)
biclustering_sec  = cca.run(data_transformed)

cca_orig = origCCA.ChengChurchAlgorithm(num_biclusters=100, msr_threshold=1200,
                                        multiple_node_deletion_threshold=1.2, data_min_cols=100)
biclustering_orig = cca_orig.run(data_transformed)

# Accuracy measurement
prelic   = accuracy.prelic_relevance(biclustering_sec, biclustering_orig)
liu_wang = accuracy.liu_wang_match_score(biclustering_sec, biclustering_orig)





