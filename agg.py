import numpy as np
import itertools
import torch
import utils.types as types
import utils.torch_tools as torch_tools

def average(vectors: list or np.ndarray) -> np.ndarray:
    """Computes the average vector in vectors.

    Argument(s):
        - vectors   : list or np.ndarray or torch.Tensor
    """

    tools = types.check_vectors(vectors)
    
    return tools.mean(vectors, axis = 0)

def median(vectors: list or np.ndarray) -> np.ndarray:
    """Computes the coordinate-wise median vector in vectors.

    Argument(s):
        - vectors: list or np.ndarray 
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")

    return np.median(vectors, axis=0)




def trmean(vectors: list or np.ndarray, nb_byz: int) -> np.ndarray:
    """Applies the Trimmed Mean aggregation rule (Yin et al. (2021)):
    Sorts the vector values by coordinates, removes the first lowest
    'nb_byz' values and the highest 'nb_byz' values by coordinates and
    applies the 'average' function to the resulting vector list.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nb_byz    : int

    Reference(s):
        - Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter 
          Bartlett. Byzantine-robust distributed learning: Towards
          optimal statistical rates. In Jennifer Dy and Andreas Krause,
          editors, Proceedings of the 35th International Conference on
          Machine Learning, volume 80 of Pro- ceedings of Machine 
          Learning Research, pages 5650–5659. PMLR, 10–15 Jul 2018. 
          URL https://proceedings.mlr.press/v80/yin18a.html.

    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")

    if nb_byz == 0:
        return average(vectors)
    
    selected_vectors = np.sort(vectors, axis = 0)[nb_byz:-nb_byz]
    return average(selected_vectors)




def geometric_median(vectors, nu=0.1, T=3):
    """Applies the smoothed Weiszfeld algorithm [XXX] to return the
    approximate geometric median vector of 'vectors'.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nu        : float
        - T         : int
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nu, float):
        raise TypeError("'nu' should be a 'float'")
    if not isinstance(T, float):
        raise TypeError("'T' should be a 'int'")

    z = np.zeros_like(vectors[0])
    filtered_vectors = vectors[~np.any(np.isinf(tab), axis = 1)]
    alpha = 1/len(vectors)
    for _ in range(T):
        betas = np.linalg.norm(filtered_vectors - z, axis = 1)
        betas[betas<nu] = nu
        betas = (alpha/betas)[:, None]
        z = (filtered_vectors*betas).sum(axis = 0) / np.sum(betas)
    return z




def krum(vectors: list or np.ndarray, nb_byz: int) -> np.ndarray:
    """Applies the Krum aggregation rule (Blanchard et al. (2017)):
    Returns the vector closest in average to 'len(vectors) - nb_byz - 1'
    other vectors.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nb_byz    : int

    Reference(s):
        - Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and 
        Julien Stainer. Machine learning with adversaries: Byzantine
        tolerant gradient descent. In I. Guyon, U. V. Luxburg,
        S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
        R. Garnett, editors, Advances in Neural Information Processing
        Systems 30, pages 119–129. Curran Associates, Inc., 2017.
        URL https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9
        f68f89b29639786cb62ef-Paper.pdf
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")
    
    dist = [[np.linalg.norm(a-b) for a in vectors] for b in vectors]
    dist = np.sort(dist)[:,1:len(vectors)-nb_byz]
    dist = np.mean(dist, axis = 1)
    index = np.argmin(dist)
    return vectors[index]




def multi_krum(vectors: list or np.ndarray, nb_byz: int) -> np.ndarray:
    """Applies the Multi-Krum function (Blanchard et al. (2017)):
    Selects the k vectors closest in average to 
    'len(vectors) - nb_byz - 1' other vectors (each vector can be the
    closest to a different subset of vectors) and returns the average
    vector of the selected vectors.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nb_byz    : int

    Reference(s):
        - Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and 
        Julien Stainer. Machine learning with adversaries: Byzantine
        tolerant gradient descent. In I. Guyon, U. V. Luxburg,
        S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
        R. Garnett, editors, Advances in Neural Information Processing
        Systems 30, pages 119–129. Curran Associates, Inc., 2017.
        URL https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9
        f68f89b29639786cb62ef-Paper.pdf
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")

    dist = [[np.linalg.norm(a - b) for a in vectors] for b in vectors]
    dist = np.sort(dist)[:,1:len(vectors)-nb_byz]
    dist = np.mean(dist, axis = 1)
    k = len(vectors) - nb_byz
    indices = np.argpartition(dist, k)[:k]
    return average(np.array(vectors)[indices])




def nnm(vectors: list or np.ndarray, nb_byz: int) -> np.ndarray:
    """Applies the Nearest Neighbor Mixing (NNM) pre-aggregation rule 
    (Allouah et al. (2023)): returns a 2 dimensionnal array of type
    'np.ndarray' containing the average of the 'len(vectors) - nb_byz'
    nearest neighbors of each vector in 'vectors'.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nb_byz    : int

    Reference(s):
        - Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui,
          Nirupam Gupta, Rafaël Pinot, and John Stephan. Fixing by
          mixing: A recipe for optimal Byzantine ML under heterogeneity.
          In International Conference on Artificial Intelligence and
          Statistics, pages 1232–1300. PMLR, 2023. URL 
          https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")

    dist = [[np.linalg.norm(a-b) for a in vectors] for b in vectors]
    k = len(vectors) - nb_byz
    indices = np.argpartition(dist, k , axis = 1)[:,:k]
    selected_vectors = np.array(vectors)[indices]
    return np.mean(selected_vectors, axis = 1)




def bucketing(vectors: list or np.ndarray, bucket_size: int) -> np.ndarray:
    """Applies the Bucketing aggregation rule (Karimireddy et al., 2022)
    Returns a 2 dimensionnal array of type 'np.ndarray' containing
    averages of 'bucket_size' vectors. Each average is computed on a
    disjoint subset of 'bucket_size' vectors drawn uniformely whithout
    replacement in 'vectors'.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nb_byz    : int

    Reference(s):
        - Karimireddy, S. P., He, L., and Jaggi, M. (2022). Byzantine-
          robust learning on heterogeneous datasets via bucketing. In
          International Conference on Learning Repre- sentations. 
          URL https://openreview.net/pdf?id=jXKKDEi5vJt
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(bucket_size, int):
        raise TypeError("'bucket_size' should be a 'int'")

    np.random.shuffle(vectors)
    nb_buckets = int(np.floor(len(vectors) / bucket_size))
    complete_buckets = vectors[:nb_buckets * bucket_size]
    complete_buckets.resize(nb_buckets, bucket_size, len(vectors[0]))
    output = np.mean(complete_buckets, axis = 1)
    
    # Adding the last incomplete bucket if it exists
    if nb_buckets != len(vectors) / bucket_size :
        last_mean = np.mean(vectors[nb_buckets * bucket_size:], axis = 0)
        output = np.concatenate((output, last_mean.resize(1,-1)), axis = 0)
    return returned_vector




def centered_clipping(vectors: list or np.ndarray, 
                      prev_momentum: np.array,
                      L_iter: int = 3, 
                      clip_thresh: int = 1) -> np.array:

    """Applies the Centered Clipping Algorithm presented in
    (Karimireddy et al.(2021)): It adds to 'prev_momentum' the average
    clipped differences between 'prev_momentum' and each of the vectors
    in 'vectors'. This is done 'L_iter' times using at each iteration
    the new value of 'prev_momentum'
    
    Argument(s):
        - vectors       : list or np.ndarray 
        - prev_momentum : np.ndarray
        - L_iter        : int
        - clip_thresh   : int

    Reference(s):
        - Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning
        from history for byzantine robust optimization. In 38th
        International Conference on Machine Learning (ICML), 2021.
        URL http://proceedings.mlr.press/v139/karimireddy21a/karimir
        eddy21a.pdf
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(prev_momentum, np.ndarray):
        raise TypeError("'prev_momentum' should be an 'np.ndarray'")
    if not isinstance(L_iter, int):
        raise TypeError("'L_iter' should be a 'int'")
    if not isinstance(clip_thresh, int):
        raise TypeError("'clip_thresh' should be a 'int'")

    v = prev_momentum
    vectors = np.array(vectors)
    for i in range(L_iter):
        differences = vectors - v
        clip_factor = clip_thresh / np.linealg.norm(differences, axis = 1)
        clip_factor = np.minimum(np.ones_like(clip_factor), clip_factor)
        differences = np.multiply(differences, clip_factor.reshape(-1,1))
        v = v + np.mean(differences, axis = 0)
    return v




def minimum_diameter_averaging(vectors: list[np.array] or np.array,
                               nb_byz: int) -> np.array:

    """Finds the subset of vectors of size 'len(vectors) - nb_byz' that
    has the smallest diameter and returns the average of the vectors in
    the selected subset. The diameter of a subset of vectors is equal
    to the maximal distance between two vectors in the subset.

    Argument(s):
        - vectors       : list or np.ndarray 
        - nb_byz        : int
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")
            
    dist = np.array([[np.linalg.norm(a-b) for a in vectors] for b in vectors])
    
    n = len(vectors)
    k = n - nb_byz

    min_diameter = np.inf
    min_subset = np.arange(k)

    all_subsets = list(itertools.combinations(range(n), k))
    for subset in all_subsets:
        vector_indeces = list(itertools.combinations(subset, 2))
        diameter = np.max(dist[tuple(zip(*vector_indeces))])
        if diameter < min_diameter:
            min_subset = subset
            
    return vectors[np.array(min_subset)].mean(axis = 0)




def minimum_variance_averaging(vectors: list[np.array] or np.array,
                               nb_byz: int) -> np.array:
    """Finds the subset of vectors of size 'len(vectors) - nb_byz' that
    has the smallest variance and returns the average of the vectors in
    the selected subset. The variance of a subset is equal to the sum
    of the difference's norm between each pair of vectors divided by 
    'len(vectors) - nb_byz'. Note that we do not apply this last 
    division since all the subsets have the same length, hence the
    comparision is the same whithout the division.

    Argument(s):
        - vectors       : list or np.ndarray 
        - nb_byz        : int
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")
            
    dist = np.array([[np.linalg.norm(a-b) for a in vectors] for b in vectors])
    
    n = len(vectors)
    k = n - nb_byz

    min_diameter = np.inf
    min_subset = np.arange(k)

    all_subsets = list(itertools.combinations(range(n), k))
    for subset in all_subsets:
        vector_indeces = list(itertools.combinations(subset, 2))
        diameter = np.sum(dist[tuple(zip(*vector_indeces))])
        if diameter < min_diameter:
            min_subset = subset
            
    return vectors[np.array(min_subset)].mean(axis = 0)




def monna(vectors: list[np.array] or np.array, 
          nb_byz: int,
          pivot_index: int = -1) -> np.array:
    """Returns the average of the 'len(vectors)-nb_byz' closest vectors
    to 'vectors[pivot_index]'.
    
    Argument(s):
        - vectors       : list or np.ndarray 
        - nb_byz        : int
        - pivot_index   : int
    """

    if not (isinstance(vectors, list) or isinstance(vectors, np.ndarray)):
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'")
    if not isinstance(nb_byz, int):
        raise TypeError("'nb_byz' should be a 'int'")

    dist = [np.linalg.norm(vectors[pivot_index]-v) for v in vectors]
    k = len(vectors) - nb_byz
    indices = np.argpartition(dist, k)[:k]
    return average(np.array(vectors)[indices])










