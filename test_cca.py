import numpy as np
import cca
import pandas as pd
from os.path import dirname, join


"""n_rows, n_cols = 20, 17
n_elements = n_rows*n_cols
# np.random.seed(42) # Fixed seed for reproducibility
# data = np.random.randint(0, 1, size=(n_rows, n_cols))
data = np.random.randint(0, 100, size=(n_rows, n_cols))"""
# print(data)


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
    # return pd.DataFrame(data)


data = load_yeast_tavazoie().values
print(data)
# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=1, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
cca = cca.ChengChurchAlgorithm(num_biclusters=10, msr_threshold=300,
                               multiple_node_deletion_threshold=1.2, data_min_cols=100)
biclustering = cca.run(data)
print(biclustering)
