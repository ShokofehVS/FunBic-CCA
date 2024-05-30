import math
import numpy as np
import cca
import origCCA
import pandas as pd
from os.path import dirname, join
from urllib.request import urlopen
import accuracy
import seaborn as sns
import matplotlib.pyplot as plt

# Numpy sample data generation
"""n_rows, n_cols = 512, 17
n_elements = n_rows*n_cols
# np.random.seed(42) # Fixed seed for reproducibility
data = np.random.randint(0, 600, size=(n_rows, n_cols)) #IT CANNOT GO BEYOND THAT DUE TO HIGH NUMBER OF BITS OF INTEGER!
print(data)"""

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


def load_human_data():
    with urlopen("http://arep.med.harvard.edu/biclustering/lymphoma.matrix") as f:
        lines = f.read().decode('utf-8').strip().split('\n')
        lines = list(' -'.join(line.split('-')).split(' ') for line in lines)

        lines = list(list(int(i) for i in line if i) for line in lines)
        data = np.array(lines)
        if lines:
            if "true" in lines:
                print("There is a profane word in the document")

        return data


# Load data sets
data = load_yeast_tavazoie().values
# data = load_human_data()

# missing value imputation suggested by Cheng and Church for Yeast data set
missing = np.where((data <= 0.0))
data[missing] = np.random.randint(low=1, high=800, size=len(missing[0]))

# missing value imputation suggested by Cheng and Church for Human data set
"""generator = np.random.RandomState(0)
idx = np.where(data == 999)
data[idx] = generator.randint(1, 800, len(idx[0]))

missing = np.where((data <= 0.0))
data[missing] = np.random.randint(low=1, high=800, size=len(missing[0]))"""

# data_transformed = (np.log(data)).astype(int) #ANOTHER DISTRIBUTION

cca = cca.ChengChurchAlgorithm(num_biclusters=100, msr_threshold=1200,
                               multiple_node_deletion_threshold=1.2, data_min_cols=100)

cca_orig = origCCA.ChengChurchAlgorithm(num_biclusters=100, msr_threshold=1200,
                                        multiple_node_deletion_threshold=1.2, data_min_cols=100)

percent = [1, 5, 10, 30, 50, 70, 90, 100]
for i in range(len(percent)):
    # Data distribution for better accuracy
    data_transformed = (data * (percent[i]/100)).astype(int)
    n_rows, n_cols = data_transformed.shape

    # Creating an instance of the ChengChurchAlgorithm class and running with the parameters of the original study
    biclustering = cca.run(data_transformed)
    biclustering_orig = cca_orig.run(data_transformed)

    # Compare both version with CE, liu and prelic match scores
    ce = accuracy.clustering_error(biclustering, biclustering_orig, n_rows, n_cols)
    # prelic = accuracy.prelic_relevance(biclustering,biclustering_orig)
    # liu_wang = accuracy.liu_wang_match_score(biclustering, biclustering_orig)

    with open('ce.csv', 'a') as saveFile:
        saveFile.write("\n")
        saveFile.write(str(ce) + "\t")
        saveFile.write("%" + str(percent[i]))
        saveFile.write("\n")





