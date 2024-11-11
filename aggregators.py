import itertools

import numpy as np
import torch

import utils.misc as misc

class Average(object):
    """
    Compute the average along the first axis
        
    Initialization parameters
    -------------------------
    None
        
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
        >>> agg = aggregators.Average()

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([4. 5. 6.])
                
        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([4., 5., 6.])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([4., 5., 6.])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([4., 5., 6.])

        ---------

    """
    def __init__(self, **kwargs):
        pass

    def aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.mean(vectors, axis=0)
    
    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)


class Median(object):

    """
    Compute the coordinate-wise median along the first axis
        

    Initialization parameters
    -------------------------
    None
        
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
        >>> agg = aggregators.Median()

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([4. 5. 6.])
                
        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([4., 5., 6.])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([4., 5., 6.])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([4., 5., 6.])
    """
    def __init__(self, **kwargs):
        pass
    
    def aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.median(vectors, axis=0)

    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)


class TrMean(object):
    r"""
    Compute the trimmed mean (or truncated mean) along the first axis:

    .. math::

        \left[\mathrm{TrMean}_{f} \ (x_1, \dots, x_n)\right]_k = \frac{1}{n - 2f}\sum_{j = f+1}^{n-f} \left[x_{\tau_k(j)}\right]_k
    
    where 
    
    - \\([\\cdot]_k\\) refers to the k-th coordinate

    - \\(\\tau_k\\) denote a permutation on \\([n]\\) that sorts the k-th
      coordinate of the input vectors in non-decreasing order, i.e., 
      \\([x_{\\tau_k(1)}]_k \\leq ...\\leq [x_{\\tau_k(n)}]_k\\)
    
    In other words, TrMean removes the \\(f\\) highest values and \\(f\\) 
    lowers values coordinate-wise, and then applies the average.

    Initialization parameters
    --------------------------
    f : int, optional
        Number of trimmed values to be considered in the aggregation. The default is setting \\(f=0\\).

    
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
        >>> agg = aggregators.TrMean(1)

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([4. 5. 6.])
                
        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([4., 5., 6.])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([4., 5., 6.])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([4., 5., 6.])


    """

    def __init__(self, f=0, **kwargs):
        self.f = f

    def aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.f, int)

        if self.f == 0:
            avg = Average()
            return avg.aggregate_vectors(vectors)

        selected_vectors = tools.sort(vectors, axis=0)[self.f:-self.f]
        return tools.mean(selected_vectors, axis=0)

    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)


class GeometricMedian(object):
    r"""
    Apply the smoothed Weiszfeld algorithm [1]_ to return the
    approximate geometric median vector:

    .. math::

        \mathrm{GeometricMedian}_{\nu, T} \ (x_1, \dots, x_n) \in \argmin_{y}\sum_{i = 1}^{n} \|y - x_i\|_2
        
    
    Initialization parameters
    --------------------------
    nu : float, optional
    T : int, optional
    
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
        >>> agg = aggregators.GeometricMedian()

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([3.78788764 4.78788764 5.78788764])

        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([3.7879, 4.7879, 5.7879])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([3.78788764 4.78788764 5.78788764])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([3.7879, 4.7879, 5.7879])


    References
    ----------

    .. [1] Endre Weiszfeld. Sur le point pour lequel la somme des distances de 
           n points donnés est minimum. Tohoku Mathematical Journal, First Series, 
           43:355–386, 1937

    """

    def __init__(self, nu=0.1, T=3, **kwargs):
        self.nu = nu
        self.T = T

    def aggregate_vectors(self, vectors):
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

    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)

class Krum(object):
    r"""
    Apply the Krum aggregation rule [1]_:

    .. math::

        \mathrm{Krum}_{f} \ (x_1, \dots, x_n) = x_{k^\star}
        
    with

    .. math::

        k^\star \in \argmin_{i \in [n]} \sum_{x \in \mathcal{N}_i} \|x_i - x\|^2_2

    where \\(\\mathcal{N}_i\\) is the set of the \\(n − f\\) nearest 
    neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\)

    
    Initialization parameters
    --------------------------
    f : int, optional
    
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
        >>> agg = aggregators.Krum(1)

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([1. 2. 3.])

        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([1., 2., 3.])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([1. 2. 3.])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of  torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([1., 2., 3.])


    References
    ----------

    .. [1] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guer- raoui, and Julien
           Stainer. Machine learning with adversaries: Byzantine tolerant 
           gradient descent. In I. Guyon, U. V. Luxburg, S. Bengio, H. 
           Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, 
           Advances in Neural Information Processing Systems 30, pages 
           119–129. Curran Associates, Inc., 2017.
    """

    def __init__(self, f=0, **kwargs):
        self.f = f
    
    def aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.f, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis=1)[:,1:len(vectors)-self.f]
        dist = tools.mean(dist, axis=1)
        print(dist)
        index = tools.argmin(dist)
        return vectors[index]

    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)

class MultiKrum(object):

    r"""
    Apply the Multi-Krum aggregation rule [1]_:

    .. math::

        \mathrm{MultiKrum}_{f} \ (x_1, \dots, x_n) = \frac{1}{n-f}\sum_{i = 1}^{n-f} x_{k^\star_i}
        
    with

    .. math::

        \sum_{x \in \mathcal{N}_{k^\star_1} } \|x_{k^\star_1} - x\|^2_2 \leq \dots \leq \sum_{x \in \mathcal{N}_{k^\star_{n-f}} } \|x_{k^\star_{n-f}} - x\|^2_2 

    where for any \\(i \\in [n], \\mathcal{N}_i\\) is the set of the \\(n − f\\) nearest 
    neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\)

    
    Initialization parameters
    --------------------------
    f : int, optional
    
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
        >>> agg = aggregators.MultiKrum(1)

        Using numpy arrays
            
        >>> import numpy as np
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([2.5 3.5 4.5])

        Using torch tensors
            
        >>> import torch
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([2.5000, 3.5000, 4.5000])

        Using list of numpy arrays

        >>> import numppy as np
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([2.5 3.5 4.5])

        Using list of torch tensors
            
        >>> import torch
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([2.5000, 3.5000, 4.5000])


    References
    ----------

    .. [1] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guer- raoui, and Julien
           Stainer. Machine learning with adversaries: Byzantine tolerant 
           gradient descent. In I. Guyon, U. V. Luxburg, S. Bengio, H. 
           Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, 
           Advances in Neural Information Processing Systems 30, pages 
           119–129. Curran Associates, Inc., 2017.
    """

    def __init__(self, f = 0, **kwargs):
        self.f = f
    
    def aggregate_vectors(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_type(self.f, int)

        distance = misc.distance_tool(vectors)

        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis = 1)[:,1:len(vectors)-self.f]
        dist = tools.mean(dist, axis = 1)
        k = len(vectors) - self.f
        indices = tools.argpartition(dist, k-1)[:k]
        return tools.mean(vectors[indices], axis=0)

    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)

class CenteredClipping(object):
    r"""
    Apply the Centered Clipping aggregation rule [1]_:

    .. math::

        \mathrm{CenteredClipping}_{m, L, \tau} \ (x_1, \dots, x_n) = v_{L}
        
    with

    .. math::

        v_0 &= m \\
        v_{l+1} &= v_{l} + \frac{1}{n}\sum_{i=1}^{n}(x_i - v_l)\min\left(1, \frac{\tau}{\|x_i - v_l\|}\right) \ \ ; \ \forall l \in \{1,\dots, L\}

    Initialization parameters
    --------------------------
    m : numpy.ndarray, torch.Tensor, optional
        Value on which the Center Clipping aggregation starts. Default makes 
        it start from a vector with all its coordinates equal to 0.
    L : int, optional
        Number of iterations. Default is set to 1.
    tau : float, optional
          Clipping threshold. Default is set to 100.

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

    Note
    ----

        If the instance is called more than once, the value of \\(m\\) used in
        the next call is equal to the output vector of the previous call.

    Note
    ----
        
        In case you specify the optional parameter \\(m\\) when initializing 
        the instance, ensure that it has the same type and shape as the input
        vectors \\(x_i\\) that you will use when calling the instance.

    Examples
    --------
        
        >>> import aggregators
        >>> import numpy as np
        >>> import torch

        Using numpy arrays

        >>> agg = aggregators.CenteredClipping()
        >>> x = np.array([[1., 2., 3.],       # np.ndarray
        >>>               [4., 5., 6.], 
        >>>               [7., 8., 9.]])
        >>> agg(x)
        array([4., 5., 6.])
        
        Using torch tensors

        >>> agg = aggregators.CenteredClipping()
        >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
        >>>                   [4., 5., 6.], 
        >>>                   [7., 8., 9.]])
        >>> agg(x)
        tensor([4., 5., 6.])

        Using list of numpy arrays

        >>> agg = aggregators.CenteredClipping()
        >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
        >>>      np.array([4., 5., 6.]), 
        >>>      np.array([7., 8., 9.])]
        >>> agg(x)
        array([4., 5., 6.])
        
        Using list of torch tensors

        >>> agg = aggregators.CenteredClipping()
        >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
        >>>      torch.tensor([4., 5., 6.]), 
        >>>      torch.tensor([7., 8., 9.])]
        >>> agg(x)
        tensor([4., 5., 6.])


    References
    ----------

    .. [1] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning
           from history for byzantine robust optimization. In 38th
           International Conference on Machine Learning (ICML), 2021.
    """

    def __init__(self, m=None, L=1, tau=100, **kwargs):
        self.m = m
        self.L = L
        self.tau = tau

    def aggregate_vectors(self, vectors):

        tools, vectors = misc.check_vectors_type(vectors)

        if self.m is None:
            self.m = tools.zeros_like(vectors[0])

        misc.check_type(self.m, (np.ndarray, torch.Tensor))
        misc.check_type(self.L, int)
        misc.check_type(self.tau, int)

        v = self.m

        for _ in range(self.L):
            differences = vectors - v
            clip_factor = self.tau / tools.linalg.norm(differences, axis = 1)
            clip_factor = tools.minimum(tools.ones_like(clip_factor), clip_factor)
            differences = tools.multiply(differences, clip_factor.reshape(-1,1))
            v = tools.add(v, tools.mean(differences, axis=0))
        
        self.m = v

        return v
    
    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)


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
    
    def __call__(self, vectors):
        return self.aggregate_vectors(vectors)


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
