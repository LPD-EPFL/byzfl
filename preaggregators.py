import math

import utils.misc as misc

class NNM(object):

    """
    Description
    -----------
    Applies the Nearest Neighbor Mixing (NNM) pre-aggregation rule (Allouah et al. (2023)): 
     Returns a 2 dimensionnal array of type "np.ndarray" containing the average 
     of the (len(vectors) - nb_byz) nearest neighbors of each vector.

    Reference(s)
    ------------
    Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui,
    Nirupam Gupta, Rafaël Pinot, and John Stephan. Fixing by
    mixing: A recipe for optimal Byzantine ML under heterogeneity.
    In International Conference on Artificial Intelligence and
    Statistics, pages 1232–1300. PMLR, 2023. 
    URL https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "pre_aggregators" : [{
    >>>     "name": "NNM",
    >>>     "parameters": {
    >>>         "nb_byz" : 2
    >>>     }
    >>> }]

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz
    
    def pre_aggregate_vectors(self, vectors):
        """
        Applies the Nearest Neighbor Mixing

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        list or np.ndarray or torch.Tensor
            Input vector with NNM applied. 
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> pre_agg = NNM(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                ndarray([[2.5 3.5 4.5]
                        [2.5 3.5 4.5]
                        [5.5 6.5 7.5]])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                tensor([[2.5000, 3.5000, 4.5000],
                        [2.5000, 3.5000, 4.5000],
                        [5.5000, 6.5000, 7.5000]])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)
        k = len(vectors) - self.nb_byz
        indices = tools.argpartition(dist, k , axis = 1)[:,:k]
        return tools.mean(vectors[indices], axis = 1)


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


class Bucketing(object):

    """
    Description
    ------------
    Applies the Bucketing aggregation rule (Karimireddy et al., 2022):
    Returns a 2 dimensionnal array of type "np.ndarray" containing
    averages of "bucket_size" vectors. Each average is computed on a
    disjoint subset of "bucket_size" vectors drawn uniformely whithout
    replacement in "vectors".

    Reference(s)
    -------------
    Karimireddy, S. P., He, L., and Jaggi, M. (2022). Byzantine-
    robust learning on heterogeneous datasets via bucketing. In
    International Conference on Learning Representations. 
    URL https://openreview.net/pdf?id=jXKKDEi5vJt

    Parameters
    -----------
    bucket_size : int 
        Size of the buckets

    How to use it in experiments
    -----------------------------
    >>> "pre_aggregators" : [{
    >>>     "name": "Bucketing",
    >>>     "parameters": {
    >>>         "bucket_size" : 2,
    >>>     }
    >>> }]

    Methods
    ---------


    """

    def __init__(self, nb_workers, nb_byz, bucket_size=None, **kwargs):
        self.bucket_size = bucket_size
        if self.bucket_size is None:
            self.bucket_size = math.floor(nb_workers/(2*nb_byz))
    
    def pre_aggregate_vectors(self, vectors):
        """
        Applies Bucketing

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        list or np.ndarray or torch.Tensor
            Input vector with Bucketing applied. 
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> pre_agg = Bucketing(2)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                ndarray([[4. 5. 6.]
                        [4. 5. 6.]])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = pre_agg.pre_aggregate_vectors(vectors)
                >>> print(result)
                tensor([[5.5000, 6.5000, 7.5000],
                        [1.0000, 2.0000, 3.0000]])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.bucket_size, int)

        random = misc.random_tool(vectors)

        vectors = random.permutation(vectors)
        nb_buckets = int(math.floor(len(vectors) / self.bucket_size))
        buckets = vectors[:nb_buckets * self.bucket_size]
        buckets = tools.reshape(buckets, (nb_buckets, self.bucket_size, len(vectors[0])))
        output = tools.mean(buckets, axis = 1)
        
        # Adding the last incomplete bucket if it exists
        if nb_buckets != len(vectors) / self.bucket_size :
            last_mean = tools.mean(vectors[nb_buckets * self.bucket_size:], axis = 0)
            last_mean = last_mean.reshape(1,-1)
            output = tools.concatenate((output, last_mean), axis = 0)
        return output


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
