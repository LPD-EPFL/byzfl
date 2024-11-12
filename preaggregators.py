import math

import utils.misc as misc

class NNM(object):

    r"""
    Apply the Nearest Neighbor Mixing pre-aggregation rule [1]_:

    .. math::

        \mathrm{NNM}_{f} \ (x_1, \dots, x_n) = \left(\frac{1}{n-f}\sum_{i\in\mathcal{N}_{1}} x_i \ \ , \ \dots \ ,\ \  \frac{1}{n-f}\sum_{i\in\mathcal{N}_{n}} x_i \right)
        
    where \\(\\mathcal{N}_i\\) is the set of the \\(n − f\\) nearest 
    neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\)

    
    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. The default is setting \\(f=0\\).
    
    Calling the instance
    --------------------

    Input parameters
    ----------------
    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.
        
    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output will be the same as the input.

    Examples
    --------

        
        >>> import aggregators
        >>> agg = preaggregators.NNM(1)

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([[2.5 3.5 4.5]
               [2.5 3.5 4.5]
               [5.5 6.5 7.5]])

        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([[2.5000, 3.5000, 4.5000],
                [2.5000, 3.5000, 4.5000],
                [5.5000, 6.5000, 7.5000]])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([[2.5 3.5 4.5]
               [2.5 3.5 4.5]
               [5.5 6.5 7.5]])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of  torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([[2.5000, 3.5000, 4.5000],
                [2.5000, 3.5000, 4.5000],
                [5.5000, 6.5000, 7.5000]])


    References
    ----------

    .. [1] Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R.,
           & Stephan, J. (2023, April). Fixing by mixing: A recipe for optimal
           byzantine ml under heterogeneity. In International Conference on 
           Artificial Intelligence and Statistics (pp. 1232-1300). PMLR.    

    """

    def __init__(self, f=0, **kwargs):
        self.f = f
    
    def pre_aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.f, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)
        k = len(vectors) - self.f
        indices = tools.argpartition(dist, k-1, axis = 1)[:,:k]
        return tools.mean(vectors[indices], axis = 1)

    def __call__(self, vectors):
        return self.pre_aggregate_vectors(vectors)

class Bucketing(object):

    r"""
    Apply the Bucketing pre-aggregation rule [1]_:

    .. math::

        \mathrm{Bucketing}_{f} \ (x_1, \dots, x_n) = 
        \left(\frac{1}{s}\sum_{i=0}^s x_{\pi(i)} \ \ , \ \ 
        \frac{1}{s}\sum_{i=s+1}^{2s} x_{\pi(i)} \ \ , \ \dots \ ,\ \  
        \frac{1}{s}\sum_{i=\left(\lceil n/s \rceil-1\right)s+1}^{n} x_{\pi(i)} \right)

    where \\(\\pi\\) is a random permutation of  \\([n]\\).

    Initialization parameters
    --------------------------
    s : int, optional
        Number of vectors per bucket. The default is setting \\(n=1\\).
    
    Calling the instance
    --------------------

    Input parameters
    ----------------
    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.
        
    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output will be the same as the input.

    Examples
    --------

        
        >>> import aggregators
        >>> agg = preaggregators.Bucketing(2)

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([[4. 5. 6.]
               [4. 5. 6.]])

        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([[5.5000, 6.5000, 7.5000],
                [1.0000, 2.0000, 3.0000]])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([[4. 5. 6.]
               [4. 5. 6.]])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of  torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([[5.5000, 6.5000, 7.5000],
                [1.0000, 2.0000, 3.0000]])

        
    Note
    ----
        
    The results when using torch tensor and numpy array differ as it 
    depends on random permutation that are not necessary the same


    References
    ----------

    .. [1] Karimireddy, S. P., He, L., & Jaggi, M. (2020). Byzantine-robust 
           learning on heterogeneous datasets via bucketing. International 
           Conference on Learning Representations 2022.
    """

    def __init__(self, s=1, **kwargs):
        self.s = s
    
    def pre_aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.s, int)

        random = misc.random_tool(vectors)

        vectors = random.permutation(vectors)
        nb_buckets = int(math.floor(len(vectors) / self.s))
        buckets = vectors[:nb_buckets * self.s]
        buckets = tools.reshape(buckets, (nb_buckets, self.s, len(vectors[0])))
        output = tools.mean(buckets, axis = 1)
        
        # Adding the last incomplete bucket if it exists
        if nb_buckets != len(vectors) / self.s :
            last_mean = tools.mean(vectors[nb_buckets * self.s:], axis = 0)
            last_mean = last_mean.reshape(1,-1)
            output = tools.concatenate((output, last_mean), axis = 0)
        return output

    def __call__(self, vectors):
        return self.pre_aggregate_vectors(vectors)


class Identity(object):
    """
    Description
    -----------
    Return a copy of the same vectors

    How to use it in experiments
    ----------------------------
    >>> "pre_aggregators" : [{
    >>>     "name": "Identity",
    >>>     "parameters": {}
    >>> }]

    Methods
    ---------
    """
    def __init__(self, **kwargs):
        pass
    
    def pre_aggregate_vectors(self, vectors):
        """
        Applies Identity

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        list or np.ndarray or torch.Tensor
            Input vector with Idenity applied. 
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> pre_agg = Identity()
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                ndarray([[1. 2. 3.]
                        [4. 5. 6.]
                        [7. 8. 9.]])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                tensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.]])
        """
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.copy(vectors)
    def __call__(self, vectors):
        return self.pre_aggregate_vectors(vectors)


class Clipping(object):
    def __init__(self, clip_factor=2, **kwargs):
        self.clip_factor = clip_factor
    
    def _clip_vector(self, vector):
        tools, vector = misc.check_vectors_type(vector)
        vector_norm = tools.linalg.norm(vector)
        if vector_norm > self.clip_factor:
            vector = tools.multiply(vector, self.clip_factor / vector_norm)
        return vector
    
    def pre_aggregate_vectors(self, vectors):
        return [self._clip_vector(gradient) for gradient in vectors]

    def __call__(self, vectors):
        return self.pre_aggregate_vectors(vectors)
class Arc(object):
    """
    Description
    -----------
    Applies Arc pre-aggregation rule

    Parameters
    ----------
    nb_workers : int
        Number of workers (nodes) in the training
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "pre_aggregators" : [{
    >>>     "name": "Arc",
    >>>     "parameters": {
    >>>         "nb_workers" : 10,
    >>>         "nb_byz" : 2
    >>>     }
    >>> }]

    Methods
    ---------
    """
    def __init__(self, nb_workers, nb_byz, **kwargs):
        self.nb_workers = nb_workers
        self.nb_byz = nb_byz
    
    def _clip_vector(vector, clip_threshold):
        """
        Private function to clip a vector

        Parameters
        ----------
        vector : 1D ndarray or 1D torch.tensor
            Input vector to clip
        clip_threshold : float 
            Threshold to clip the vector

        Returns
        -------
        The same vector if it note reaches the threshold and a vector
        with the same directon but their norm now is the clip_threshold.
        """
        tools, vector = misc.check_vectors_type(vector)
        vector_norm = tools.linalg.norm(vector)
        if vector_norm > clip_threshold:
            tools.multiply(vector, (clip_threshold / vector_norm))
        return vector

    def pre_aggregate_vectors(self, vectors):
        """
        Applies the Arc pre-aggregation

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        list or np.ndarray or torch.Tensor
            Input vector with Arc applied. 
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> pre_agg = Arc(nb-workers=4, nb-byz=1})
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                ndarray([[1. 2. 3.]
                        [4. 5. 6.]
                        [7. 8. 9.]])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                tensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.]])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        magnitudes = [(tools.linalg.norm(vector), vector_id) 
                      for vector_id, vector in enumerate(vectors)]
        magnitudes.sort(key=lambda x:x[0])
        nb_clipped = int((2 * self.nb_byz / self.nb_workers) * (self.nb_workers - self.nb_byz))
        cut_off_value = self.nb_workers - nb_clipped

        f_largest = magnitudes[cut_off_value:]
        clipping_threshold = magnitudes[cut_off_value - 1][0]
        for _, vector_id in f_largest:
            vectors[vector_id] = self._clip_vector(vectors[vector_id], clipping_threshold)
        return vectors

    def __call__(self, vectors):
        return self.pre_aggregate_vectors(vectors)


