import itertools

import numpy as np
import torch

import utils.misc as misc

class Average(object):
    """
    Description
    -----------
    Aggregator that computes the average vector in vectors.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "Average",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """
    def __init__(self, **kwargs):
        pass

    def aggregate_vectors(self, vectors):
        """
        Computes the arithmetic mean along axis=0.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            The average vector of the input. The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = Average()
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([4. 5. 6.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([4., 5., 6.])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.mean(vectors, axis=0)


class Median(object):

    """
    Description
    -----------
    Computes the coordinate-wise median vector in vectors.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "Median",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """
    def __init__(self, **kwargs):
        pass
    
    def aggregate_vectors(self, vectors):
        """
        Computes the arithmetic mean along axis=0.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            The median vector of the input. The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = Median()
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([4. 5. 6.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([4., 5., 6.])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.median(vectors, axis=0)


class TrMean(object):
    """
    Description
    -----------
    Applies the Trimmed Mean aggregation rule (Yin et al. (2021)):
     Sorts the vector values by coordinates, removes the first lowest
     'nb_byz' values and the highest 'nb_byz' values by coordinates and
     applies the 'average' function to the resulting vector list.

    Reference(s)
    ------------
    Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter 
    Bartlett. Byzantine-robust distributed learning: Towards
    optimal statistical rates. In Jennifer Dy and Andreas Krause,
    editors, Proceedings of the 35th International Conference on
    Machine Learning, volume 80 of Proceedings of Machine 
    Learning Research, pages 5650–5659. PMLR, 10–15 Jul 2018. 
     URL https://proceedings.mlr.press/v80/yin18a.html.

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "TrMean",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz

    def aggregate_vectors(self, vectors):
        """
        Sorts the vector values by coordinates, removes the first lowest
        'nb_byz' values and the highest 'nb_byz' values by coordinates and
        applies the 'average' function to the resulting vector list.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Input vector with TrMean applied. 
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = TrMean(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([4. 5. 6.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([4., 5., 6.])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        if self.nb_byz == 0:
            avg = Average()
            return avg.aggregate_vectors(vectors)

        selected_vectors = tools.sort(vectors, axis=0)[self.nb_byz:-self.nb_byz]
        return tools.mean(selected_vectors, axis=0)


class GeometricMedian(object):

    """
    Description
    -----------
    Applies the smoothed Weiszfeld algorithm [XXX] to return the
    approximate geometric median vector.

    Parameters
    ----------
    nu : float
    T : int

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "GeometricMedian",
    >>>     "parameters": {
    >>>         "nu": 0.1,
    >>>         "T": 1
    >>>     }
    >>> }

    Methods
    ---------
    """

    def __init__(self, nu=0.1, T=3, **kwargs):
        self.nu = nu
        self.T = T

    def aggregate_vectors(self, vectors):
        """
        Applies the smoothed Weiszfeld algorithm [XXX] to return the
        approximate geometric median vector of 'vectors'.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Return the approximate geometric median vector
            using smoothed Weiszfeld algorithm.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = GeometricMedian(nu=1., T=3})
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([3.78788764 4.78788764 5.78788764])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([3.7879, 4.7879, 5.7879])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nu, float)
        misc.check_type(self.T, int)

        z = tools.zeros_like(vectors[0])
        filtered_vectors = vectors[~tools.any(tools.isinf(vectors), axis = 1)]
        alpha = 1/len(vectors)
        for _ in range(self.T):
            betas = tools.linalg.norm(filtered_vectors - z, axis = 1)
            betas[betas<self.nu] = self.nu
            betas = (alpha/betas)[:, None]
            z = tools.sum((filtered_vectors*betas), axis=0) / tools.sum(betas)
        return z

class Krum(object):

    """
    Description
    -----------
    Applies the Krum aggregation rule (Blanchard et al. (2017)):
     Returns the vector closest in average to len(vectors) - nb_byz - 1
     other vectors.

    Reference(s)
    ------------
    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and 
    Julien Stainer. Machine learning with adversaries: Byzantine
    tolerant gradient descent. In I. Guyon, U. V. Luxburg,
    S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
    R. Garnett, editors, Advances in Neural Information Processing
    Systems 30, pages 119–129. Curran Associates, Inc., 2017.
     URL https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "Krum",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz
    
    def aggregate_vectors(self, vectors):
        """
        Applies the Krum aggregation rule (Blanchard et al. (2017))

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns the vector closest in average 
            to len(vectors) - nb_byz - 1 other vectors.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = Krum(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([1. 2. 3.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([1., 2., 3.])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis=1)[:,1:len(vectors)-self.nb_byz]
        dist = tools.mean(dist, axis=1)
        index = tools.argmin(dist)
        return vectors[index]

class MultiKrum(object):
    """
    Description
    -----------
    Applies the Multi-Krum function (Blanchard et al. (2017)):
     Selects the k vectors closest in average to 
     len(vectors) - nb_byz - 1 other vectors (each vector can be the
     closest to a different subset of vectors) and returns the average
     vector of the selected vectors.

    Reference(s)
    ------------
    Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and 
    Julien Stainer. Machine learning with adversaries: Byzantine
    tolerant gradient descent. In I. Guyon, U. V. Luxburg,
    S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and
    R. Garnett, editors, Advances in Neural Information Processing
    Systems 30, pages 119–129. Curran Associates, Inc., 2017.
     URL https://proceedings.neurips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "MultiKrum",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz
    
    def aggregate_vectors(self, vectors):
        """
        Applies the Multi-Krum function (Blanchard et al. (2017)):
        Selects the k vectors closest in average to 
        len(vectors) - nb_byz - 1 other vectors (each vector can be the
        closest to a different subset of vectors) and returns the average
        vector of the selected vectors.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Selects the k vectors closest in average to 
            len(vectors) - nb_byz - 1 other vectors (each vector can be the
            closest to a different subset of vectors) and returns the average
            vector of the selected vectors.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = MultiKrum(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([2.5 3.5 4.5])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([2.5000, 3.5000, 4.5000])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis = 1)[:,1:len(vectors)-self.nb_byz]
        dist = tools.mean(dist, axis = 1)
        k = len(vectors) - self.nb_byz
        indices = tools.argpartition(dist, k)[:k]
        return tools.mean(vectors[indices], axis=0)


class CenteredClipping(object):

    """
    Description
    -----------
    Applies the Centered Clipping Algorithm presented in (Karimireddy et al.(2021)): 
     It adds to 'prev_momentum' the average clipped differences between 'prev_momentum' 
     and each of the vectors in 'vectors'. This is done 'L_iter' times using at 
     each iteration the new value of 'prev_momentum'

    Reference(s)
    ------------
    Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning
    from history for byzantine robust optimization. In 38th
    International Conference on Machine Learning (ICML), 2021.
        URL http://proceedings.mlr.press/v139/karimireddy21a/karimireddy21a.pdf

    Parameters
    ----------
    prev_momentum (np.ndarray):
    L_iter (int):
    clip_thresh (int):

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "CenteredClipping",
    >>>     "parameters": {
    >>>         "L_iter": 10,
    >>>         "clip_thresh": 2
    >>>     }
    >>> }

    Methods
    ---------
    """

    def __init__(self, previous_momentum=None, L_iter=1, clip_thresh=100, **kwargs):
        self.prev_momentum = previous_momentum
        self.L_iter = L_iter
        self.clip_thresh = clip_thresh

    def aggregate_vectors(self, vectors):
        """
        Applies the Centered Clipping Algorithm presented in
        (Karimireddy et al.(2021)): It adds to prev_momentum the average
        clipped differences between prev_momentum and each of the vectors
        in vectors. This is done L_iter times using at each iteration
        the new value of prev_momentum.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns the input vector with Centered Clipping Algorithm
            applied.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> previous_momentum = [1., 2., 3.]
                >>> L_iter = 10,
                >>> clip_thresh = 2
                >>> agg = CenteredClipping(previous_momentum=previous_momentum,
                                           L_iter=L_iter,
                                           clip_thresh=clip_thresh)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([3.91684398 4.91684398 5.91684398])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> agg = CenteredClipping(previous_momentum=previous_momentum,
                >>>                        L_iter=L_iter,
                >>>                        clip_thresh=clip_thresh)
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([[3.9168, 4.9168, 5.9168]])
         """

        tools, vectors = misc.check_vectors_type(vectors)

        if self.prev_momentum is None:
            self.prev_momentum = tools.zeros_like(vectors[0])

        misc.check_type(self.prev_momentum, (list, np.ndarray, torch.Tensor, int))
        misc.check_type(self.L_iter, int)
        misc.check_type(self.clip_thresh, int)

        v = self.prev_momentum

        for _ in range(self.L_iter):
            differences = vectors - v
            clip_factor = self.clip_thresh / tools.linalg.norm(differences, axis = 1)
            clip_factor = tools.minimum(tools.ones_like(clip_factor), clip_factor)
            differences = tools.multiply(differences, clip_factor.reshape(-1,1))
            v = tools.add(v, tools.mean(differences, axis=0))
        
        self.prev_momentum = v

        return v

class MDA(object):

    """
    Description
    -----------
    Finds the subset of vectors of size len(vectors) - nb_byz that
    has the smallest diameter and returns the average of the vectors in
    the selected subset. The diameter of a subset of vectors is equal
    to the maximal distance between two vectors in the subset.

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "MDA",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz

    def aggregate_vectors(self, vectors):
        """
        Finds the subset of vectors of size (len(vectors) - nb_byz) that
        has the smallest diameter and returns the average of the vectors in
        the selected subset. The diameter of a subset of vectors is equal
        to the maximal distance between two vectors in the subset.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor:
            Returns the mean of the subset of vectors of size (len(vectors) - nb_byz)
            that has the smallest diameter and returns the average of the vectors in
            the selected subset. The diameter of a subset of vectors is equal
            to the maximal distance between two vectors in the subset.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = MDA(2)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([1. 2. 3.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([1., 2., 3.])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)
        
        n = len(vectors)
        k = n - self.nb_byz

        min_diameter = np.inf
        min_subset = np.arange(k)

        all_subsets = list(itertools.combinations(range(n), k))
        for subset in all_subsets:
            vector_indices = list(itertools.combinations(subset, 2))
            diameter = tools.max(dist[tuple(zip(*vector_indices))])
            if diameter < min_diameter:
                min_subset = subset
                min_diameter = diameter
        return vectors[tools.asarray(min_subset)].mean(axis=0)


class MVA(object):

    """
    Description
    -----------
    Finds the subset of vectors of size (len(vectors) - nb_byz) that
    has the smallest variance and returns the average of the vectors in
    the selected subset. The variance of a subset is equal to the sum
    of the difference's norm between each pair of vectors divided by 
    (len(vectors) - nb_byz). Note that we do not apply this last 
    division since all the subsets have the same length, hence the
    comparision is the same whithout the division.

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "MVA",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """

    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz
    
    def aggregate_vectors(self, vectors):
        """
        Finds the subset of vectors of size (len(vectors) - nb_byz) that
        has the smallest variance and returns the average of the vectors in
        the selected subset. The variance of a subset is equal to the sum
        of the difference's norm between each pair of vectors divided by 
        (len(vectors) - nb_byz). Note that we do not apply this last 
        division since all the subsets have the same length, hence the
        comparision is the same whithout the division.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns the mean of the subset of vectors of size (len(vectors) - nb_byz) that
            has the smallest variance and returns the average of the vectors in
            the selected subset. The variance of a subset is equal to the sum
            of the difference's norm between each pair of vectors divided by 
            (len(vectors) - nb_byz). Note that we do not apply this last 
            division since all the subsets have the same length, hence the
            comparision is the same whithout the division.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = MVA(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([2.5 3.5 4.5])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([2.5000, 3.5000, 4.5000])
         """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)**2
        
        n = len(vectors)
        k = n - self.nb_byz

        min_diameter = np.inf
        min_subset = np.arange(k)

        all_subsets = list(itertools.combinations(range(n), k))
        for subset in all_subsets:
            vector_indices = list(itertools.combinations(subset, 2))
            diameter = tools.sum(dist[tuple(zip(*vector_indices))])
            if diameter < min_diameter:
                min_subset = subset
                min_diameter = diameter
                
        return vectors[tools.asarray(min_subset)].mean(axis=0)


class Monna(object):

    """
    Description
    -----------
    Returns the average of the (len(vectors)-nb_byz) closest vectors
    to "vectors[pivot_index]".

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.
    pivot_index : int
        Index to the pivot

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "Monna",
    >>>     "parameters": {
    >>>         "pivot_index": 2
    >>>     }
    >>> }

    Methods
    ---------
    """
    
    def __init__(self, nb_byz, pivot_index=0, **kwargs):
        self.nb_byz = nb_byz
        self.pivot_index = pivot_index
    
    def aggregate_vectors(self, vectors):
        """
        Finds the subset of vectors of size (len(vectors) - nb_byz) that
        has the smallest variance and returns the average of the vectors in
        the selected subset. The variance of a subset is equal to the sum
        of the difference's norm between each pair of vectors divided by 
        (len(vectors) - nb_by). Note that we do not apply this last 
        division since all the subsets have the same length, hence the
        comparision is the same whithout the division.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns the mean of the subset of vectors of size (len(vectors) - nb_byz) that
            has the smallest variance and returns the average of the vectors in
            the selected subset. The variance of a subset is equal to the sum
            of the difference's norm between each pair of vectors divided by 
            (len(vectors) - nb_byz). Note that we do not apply this last 
            division since all the subsets have the same length, hence the
            comparision is the same whithout the division.
            The data type of the output will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> agg = Monna(1, pivot_index=1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                     [4., 5., 6.], 
                >>>                     [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([2.5 3.5 4.5])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([2.5000, 3.5000, 4.5000])
        """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors[self.pivot_index].reshape(1,-1))
        k = len(vectors) - self.nb_byz
        indices = tools.argpartition(dist.reshape(-1), k)[:k]
        return tools.mean(vectors[indices], axis=0)


class Meamed(object):
    """
    Description
    -----------
    Implements the Meamed aggregation

    Parameters
    ----------
    nb_byz : int
        Number of byzantine nodes to be considered in the aggregation.

    How to use it in experiments
    ----------------------------
    >>> "aggregator": {
    >>>     "name": "Meamed",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """
    def __init__(self, nb_byz, **kwargs):
        self.nb_byz = nb_byz
    
    def aggregate_vectors(self, vectors):
        """
        Apply Meamed aggregation

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) 
            where each row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns a vector with the Meamed applied to the input vectors

        Examples
        --------
            With numpy arrays:
                >>> agg = Meamed(1)
                >>> vectors = np.array([[1., 2., 3.], 
                >>>                        [4., 5., 6.], 
                >>>                        [7., 8., 9.]])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                ndarray([2.5 3.5 4.5])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> vectors = torch.stack([torch.tensor([1., 2., 3.]), 
                >>>                           torch.tensor([4., 5., 6.]), 
                >>>                           torch.tensor([7., 8., 9.])])
                >>> result = agg.aggregate_vectors(vectors)
                >>> print(result)
                tensor([2.5000, 3.5000, 4.5000])
        """
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.nb_byz, int)

        d = len(vectors[0])
        k = len(vectors) - self.nb_byz

        median = tools.median(vectors, axis=0)
        abs_diff = tools.abs((vectors - median))

        indices = tools.argpartition(abs_diff, k, axis=0)[:k]
        indices = tools.multiply(indices, d)
        a = tools.arange(d)
        if not tools == np:
            a = a.to(indices.device)
        indices = tools.add(indices, a)
        return tools.mean(vectors.take(indices), axis=0)