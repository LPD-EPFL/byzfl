import numpy as np
import itertools
import torch
import utils.misc as misc
import math
from typing import Union

def average(vectors):

    """Computes the average vector in vectors.

    Argument(s):
    ------------
    vectors   : list or np.ndarray or torch.Tensor
    """

    tools, vectors = misc.check_vectors_type(vectors)

    return tools.mean(vectors, axis = 0)




def median(vectors):

    """Computes the coordinate-wise median vector in vectors.

    Argument(s):
        - vectors: list or np.ndarray 
    """

    tools, vectors = misc.check_vectors_type(vectors)

    return tools.median(vectors, axis=0)




def trmean(vectors, nb_byz) -> Union[np.ndarray, torch.Tensor]:

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
          Machine Learning, volume 80 of Proceedings of Machine 
          Learning Research, pages 5650–5659. PMLR, 10–15 Jul 2018. 
          URL https://proceedings.mlr.press/v80/yin18a.html.

    """

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    if nb_byz == 0:
        return average(vectors)

    selected_vectors = tools.sort(vectors, axis = 0)[nb_byz:-nb_byz]
    return tools.mean(selected_vectors, axis = 0)




def geometric_median(vectors, nu=0.1, T=3):

    """Applies the smoothed Weiszfeld algorithm [XXX] to return the
    approximate geometric median vector of 'vectors'.

    Argument(s):
        - vectors   : list or np.ndarray 
        - nu        : float
        - T         : int
    """
    
    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nu, float)
    misc.check_type(T, int)

    z = tools.zeros_like(vectors[0])
    filtered_vectors = vectors[~tools.any(tools.isinf(vectors), axis = 1)]
    alpha = 1/len(vectors)
    for _ in range(T):
        betas = tools.linalg.norm(filtered_vectors - z, axis = 1)
        betas[betas<nu] = nu
        betas = (alpha/betas)[:, None]
        z = tools.sum((filtered_vectors*betas), axis = 0) / tools.sum(betas)
    return z




def krum(vectors, nb_byz):

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

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    
    dist = tools.array([tools.linalg.norm(vectors-a, axis=1) for a in vectors])
    dist = tools.sort(dist, axis = 1)[:,1:len(vectors)-nb_byz]
    dist = tools.mean(dist, axis = 1)
    index = tools.argmin(dist)
    return vectors[index]




def multi_krum(vectors, nb_byz) -> Union[np.ndarray, torch.Tensor]:

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

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    dist = tools.array([tools.linalg.norm(vectors-a, axis=1) for a in vectors])
    dist = tools.sort(dist, axis = 1)[:,1:len(vectors)-nb_byz]
    dist = tools.mean(dist, axis = 1)
    k = len(vectors) - nb_byz
    indices = tools.argpartition(dist, k)[:k]
    return tools.mean(vectors[indices], axis = 0)




def nnm(vectors, nb_byz):

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

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    dist = tools.array([tools.linalg.norm(vectors-a, axis=1) for a in vectors])
    k = len(vectors) - nb_byz
    indices = tools.argpartition(dist, k , axis = 1)[:,:k]
    return tools.mean(vectors[indices], axis = 1)




def bucketing(vectors, bucket_size):

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
    
    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(bucket_size, int)

    random = misc.random_tool(vectors)

    vectors = random.permutation(vectors)
    nb_buckets = int(math.floor(len(vectors) / bucket_size))
    buckets = vectors[:nb_buckets * bucket_size]
    buckets = tools.reshape(buckets, (nb_buckets, bucket_size, len(vectors[0])))
    output = tools.mean(buckets, axis = 1)
    
    # Adding the last incomplete bucket if it exists
    if nb_buckets != len(vectors) / bucket_size :
        last_mean = tools.mean(vectors[nb_buckets * bucket_size:], axis = 0)
        last_mean = last_mean.reshape(1,-1)
        output = tools.concatenate((output, last_mean), axis = 0)
    return output




def centered_clipping(vectors, prev_momentum, L_iter=3, clip_thresh=1):

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

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(prev_momentum, (list, np.ndarray, torch.Tensor))
    misc.check_type(L_iter, int)
    misc.check_type(clip_thresh, int)

    v = prev_momentum
    for i in range(L_iter):
        differences = vectors - v
        clip_factor = clip_thresh / tools.linalg.norm(differences, axis = 1)
        clip_factor = tools.minimum(tools.ones_like(clip_factor), clip_factor)
        differences = tools.multiply(differences, clip_factor.reshape(-1,1))
        v = v + tools.mean(differences, axis = 0)
    return v




def mda(vectors, nb_byz):

    """Finds the subset of vectors of size 'len(vectors) - nb_byz' that
    has the smallest diameter and returns the average of the vectors in
    the selected subset. The diameter of a subset of vectors is equal
    to the maximal distance between two vectors in the subset.

    Argument(s):
        - vectors       : list or np.ndarray 
        - nb_byz        : int
    """
    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    dist = tools.array([tools.linalg.norm(vectors-a, axis=1) for a in vectors])
    
    n = len(vectors)
    k = n - nb_byz

    min_diameter = np.inf
    min_subset = np.arange(k)

    all_subsets = list(itertools.combinations(range(n), k))
    for subset in all_subsets:
        vector_indices = list(itertools.combinations(subset, 2))
        # print(tuple(zip(*vector_indices)))
        # print(dist)
        # print(dist[tuple(zip(*vector_indices))])
        diameter = tools.max(dist[tuple(zip(*vector_indices))])
        if diameter < min_diameter:
            min_subset = subset
        # print(min_subset)
    return vectors[tools.asarray(min_subset)].mean(axis = 0)




def mva(vectors, nb_byz):

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

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)

    dist = tools.array([tools.linalg.norm(vectors-a, axis=1) for a in vectors])
    
    n = len(vectors)
    k = n - nb_byz

    min_diameter = np.inf
    min_subset = np.arange(k)

    all_subsets = list(itertools.combinations(range(n), k))
    for subset in all_subsets:
        vector_indices = list(itertools.combinations(subset, 2))
        diameter = tools.sum(dist[tuple(zip(*vector_indices))])
        if diameter < min_diameter:
            min_subset = subset
            
    return vectors[tools.asarray(min_subset)].mean(axis = 0)




def monna(vectors, nb_byz, pivot_index=-1):

    """Returns the average of the 'len(vectors)-nb_byz' closest vectors
    to 'vectors[pivot_index]'.
    
    Argument(s):
        - vectors       : list or np.ndarray 
        - nb_byz        : int
        - pivot_index   : int
    """

    tools, vectors = misc.check_vectors_type(vectors)
    misc.check_type(nb_byz, int)


    dist = tools.linalg.norm(vectors[pivot_index]-vectors, axis = 1)
    k = len(vectors) - nb_byz
    indices = np.argpartition(dist, k)[:k]
    return average(vectors[indices])










