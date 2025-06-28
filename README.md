# FunBic-CCA

### Description
**Fun**ction Secret Sharing for **Bic**lusterings - **C**heng and **C**hurch **A**lgorithm: privacy-preserving gene expression data analysis by biclustering algorithm namely Cheng and Church Algorithm using secure Multiparty Computation (MPC) schemes, including additive secret sharing and function secret sharing in Python under the MIT license.

### Dependencies
We apply:  
1. [Funshade](https://github.com/ibarrond/funshade) for IC (Interval Containment) gate of function secret sharing scheme in 2PC protocol to perform comparison
2. [Sycret](https://github.com/OpenMined/sycret) for DPF (Distributed Point Function) gate of function secret sharing scheme in 2PC protocol to perform equality check

## External Evaluation Measure
To measure the similarity of encrypted biclusters with non-encrypted version, we use Liu Wang match score, along with Prelic relevance as external evaluation measures.

## Important Project Contents
- `cca.py` contains implementation of secured CCA utilising MPC schemes 
- `origCCA.py` contains implementation of original CCA
- `accuracy.py` contains implementation of accuracy measures
- `test_cca.py` contains sample implementation of both secured and original algorithms, gene expression data sets and evaluation measures

### Code Author and Contributor
Shokofeh VahidianSadegh, and Alberto Ibarrondo

_The code accompanying a [paper](https://www.scitepress.org/publishedPapers/2025/134554/pdf/index.html) published at SECRYPT 2025._




