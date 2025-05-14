import itertools
import numpy as np
import torch
from byzfl.utils.misc import check_vectors_type, distance_tool, shape, ones_vector, random_tool

class Average(object):

    r"""
    Description
    -----------

    Compute the average along the first axis:

    .. math::

        \mathrm{Average} (x_1, \dots, x_n) = \frac{1}{n} \sum_{j = 1}^{n} x_j

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

        
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

    >>> import byzfl
    >>> agg = byzfl.Average()

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

    >>> import numpy as np
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

    def __init__(self):
        pass        
    
    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        return tools.mean(vectors, axis=0)


class Median(object):
    
    r"""
    Description
    -----------

    Compute the coordinate-wise median along the first axis [1]_:

    .. math::

        \big[\mathrm{Median} \ (x_1, \dots, x_n)\big]_k = \mathrm{median} \big(\big[x_1\big]_k, \dots, \big[x_n\big]_k\big)

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - \\(\\big[\\cdot\\big]_k\\) refers to the \\(k\\)-th coordinate.

    - :math:`\mathrm{median}` refers to the median of :math:`n` scalars.


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

    >>> import byzfl
    >>> agg = byzfl.Median()

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

    >>> import numpy as np
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

     References
    ----------

    .. [1] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed
           learning: Towards optimal statistical rates. In International Conference on Machine Learning, pp.5650–5659. PMLR, 2018.

    """

    def __init__(self):
        pass

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        return tools.median(vectors, axis=0)


class TrMean(object):
    
    r"""
    Description
    -----------

    Compute the trimmed mean (or truncated mean) along the first axis [1]_:

    .. math::

        \big[\mathrm{TrMean}_{f} \ (x_1, \dots, x_n)\big]_k = \frac{1}{n - 2f}\sum_{j = f+1}^{n-f} \big[x_{\pi(j)}\big]_k
    
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.
    
    - \\(\\big[\\cdot\\big]_k\\) refers to the \\(k\\)-th coordinate.

    - \\(\\pi\\) denotes a permutation on \\(\\big[n\\big]\\) that sorts the \\(k\\)-th
      coordinate of the input vectors in non-decreasing order, i.e., 
      \\(\\big[x_{\\pi(1)}\\big]_k \\leq ...\\leq \\big[x_{\\pi(n)}\\big]_k\\).
    
    In other words, TrMean removes the \\(f\\) largest and \\(f\\) smallest coordinates per dimension, and then applies the average over the remaining coordinates.

    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.
    
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

    >>> import byzfl
    >>> agg = byzfl.TrMean(1)

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

    >>> import numpy as np
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


    References
    ----------

    .. [1] Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed
           learning: Towards optimal statistical rates. In International Conference on Machine Learning, pp.5650–5659. PMLR, 2018.

    """

    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")

        if self.f == 0:
            avg = Average()
            return avg(vectors)
        selected_vectors = tools.sort(vectors, axis=0)[self.f:-self.f]
        return tools.mean(selected_vectors, axis=0)


class GeometricMedian(object):
    
    r"""
    Description
    -----------

    Apply the smoothed Weiszfeld algorithm [1]_ to obtain the approximate geometric median \\(y\\):

    .. math::

        \mathrm{GeometricMedian}_{\nu, T} \ (x_1, \dots, x_n) \in \argmin_{y \in \mathbb{R}^d}\sum_{i = 1}^{n} \big|\big|y - x_i\big|\big|_2
    
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - :math:`d` is the dimensionality of the input space, i.e., :math:`d` is the number of coordinates of vectors :math:`x_1, \dots, x_n`.

    
    Initialization parameters
    --------------------------
    nu : float, optional
        Smoothing parameter. Set to 0.1 by default.
    T : int, optional
         Number of iterations of the smoothed Weiszfeld algorithm. Set to 3 by default.
    
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
        
    >>> import byzfl
    >>> agg = byzfl.GeometricMedian()

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

    >>> import numpy as np
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

    def __init__(self, nu=0.1, T=3):
        if not isinstance(nu, float):
            raise TypeError("f must be a float")
        self.nu = nu
        if not isinstance(T, int) or T < 0:
            raise ValueError("T must be a non-negative integer")
        self.T = T

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
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
    
    r"""
    Description
    -----------

    Apply the Krum aggregator [1]_:

    .. math::

        \mathrm{Krum}_{f} \ (x_1, \dots, x_n) = x_{\lambda}
        
    with

    .. math::

        \lambda \in \argmin_{i \in \big[n\big]} \sum_{x \in \mathit{N}_i} \big|\big|x_i - x\big|\big|^2_2

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.
    
    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.
    
    - For any \\(i \\in \\big[n\\big]\\), \\(\\mathit{N}_i\\) is the set of the \\(n − f\\) nearest neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\).

    
    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.
    
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
        
    >>> import byzfl
    >>> agg = byzfl.Krum(1)

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

    >>> import numpy as np
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

    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f
    
    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")
        distance = distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis=1)[:,1:n-self.f]
        dist = tools.mean(dist, axis=1)
        index = tools.argmin(dist)
        return vectors[index]


class MultiKrum(object):

    r"""
    Description
    -----------

    Apply the Multi-Krum aggregator [1]_:

    .. math::

        \mathrm{MultiKrum}_{f} \ (x_1, \dots, x_n) = \frac{1}{n-f}\sum_{i = 1}^{n-f} x_{\pi(i)}
        
    where
    
    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - For any \\(i \\in \\big[n\\big]\\), \\(\\mathit{N}_i\\) is the set of the \\(n - f\\) nearest neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\).

    - \\(\\pi\\) denotes a permutation on \\(\\big[n\\big]\\) that sorts the input vectors in non-decreasing order of squared distance to their :math:`n-f` nearest neighbors. This sorting is expressed as:

    .. math:: \sum_{x \in \mathit{N}_{\pi(1)}} \big|\big|x_{\pi(1)} - x\big|\big|_2^2 \leq \dots \leq \sum_{x \in \mathit{N}_{\pi(n)}} \big|\big|x_{\pi(n)} - x\big|\big|_2^2

    
    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.
    
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
        
    >>> import byzfl
    >>> agg = byzfl.MultiKrum(1)

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

    >>> import numpy as np
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

    .. [1] Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien
           Stainer. Machine learning with adversaries: Byzantine tolerant 
           gradient descent. In I. Guyon, U. V. Luxburg, S. Bengio, H. 
           Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, 
           Advances in Neural Information Processing Systems 30, pages 
           119–129. Curran Associates, Inc., 2017.
    """

    def __init__(self, f = 0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")
        distance = distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis = 1)[:,1:n-self.f]
        dist = tools.mean(dist, axis = 1)
        k = n - self.f
        indices = tools.argpartition(dist, k-1)[:k]
        return tools.mean(vectors[indices], axis=0)


class CenteredClipping(object):
    
    r"""
    Description
    -----------

    Apply the Centered Clipping aggregator [1]_:

    .. math::

        \mathrm{CenteredClipping}_{m, L, \tau} \ (x_1, \dots, x_n) = v_{L}
        
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - :math:`v_0 = m`.

    - :math:`v_{l+1} = v_{l} + \frac{1}{n}\sum_{i=1}^{n}(x_i - v_l)\min\left(1, \frac{\tau}{\big|\big|x_i - v_l\big|\big|_2}\right) \ \ ; \ \forall l \in \{0,\dots, L-1\}`.

    Initialization parameters
    --------------------------
    m : numpy.ndarray, torch.Tensor, optional
        Initial value of the CenteredClipping aggregator.
        Default (None) makes it start from zero, a vector with all its coordinates equal to 0.
    L : int, optional
        Number of iterations. Default is set to 1.
    tau : float, optional
          Clipping threshold. Default is set to 100.0.

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
        
        In case the optional parameter \\(m\\) is specified when initializing 
        the instance, \\(m\\) has to be of the same type and shape as the input
        vectors \\(\\{x_1, \\dots, x_n\\}\\) used when calling the instance.

    Examples
    --------
        
    >>> import byzfl
    >>> agg = byzfl.CenteredClipping()

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([4., 5., 6.])
    
    Using torch tensors
    
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([4., 5., 6.])

    Using list of numpy arrays

    >>> import numpy as np
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

    References
    ----------

    .. [1] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning
           from history for byzantine robust optimization. In 38th
           International Conference on Machine Learning (ICML), 2021.
    """

    def __init__(self, m=None, L=1, tau=100.0):
        if m is not None and (not isinstance(m, np.ndarray) or not isinstance(m, torch.Tensor)):
            raise TypeError("m must be of type np.ndarray or torch.Tensor")
        self.m = m
        if not isinstance(L, int) or L < 0:
            raise ValueError("L must be a non-negative integer")
        self.L = L
        if not isinstance(tau, float) or tau < 0.:
            raise ValueError("tau must be a non-negative float")
        self.tau = tau

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)

        if self.m is None:
            self.m = tools.zeros_like(vectors[0])
        v = self.m
        for _ in range(self.L):
            differences = vectors - v
            clip_factor = self.tau / tools.linalg.norm(differences, axis = 1)
            clip_factor = tools.minimum(tools.ones_like(clip_factor), clip_factor)
            differences = tools.multiply(differences, clip_factor.reshape(-1,1))
            v = tools.add(v, tools.mean(differences, axis=0))
        self.m = v
        return v


class MDA(object):
    
    r"""
    Description
    -----------

    Apply the Minimum Diameter Averaging aggregator [1]_:

    .. math::

        \mathrm{MDA}_{f} \ (x_1, \dots, x_n) = \frac{1}{n-f} \sum_{i\in S^\star} x_i
        
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - .. math:: S^\star \in \argmin_{\substack{S \subset \{1,\dots,n\} \\ |S|=n-f}} \left\{\max_{i,j \in S} \big|\big|x_i - x_j\big|\big|_2\right\}.
    
    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.

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
        
    >>> import byzfl
    >>> agg = byzfl.MDA(1)

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([2.5, 3.5, 4.5])

    Using torch tensors

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([2.5, 3.5, 4.5])
    
    Using list of torch tensors

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])


    References
    ----------

    .. [1] El Mhamdi, E. M., Guerraoui, R., Guirguis, A., Hoang, L. N., and 
           Rouault, S. Genuinely distributed byzantine machine learning. In 
           Proceedings of the 39th Symposium on Principles of Distributed 
           Computing, pp. 355–364, 2020.

    """

    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f
    
    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")

        distance = distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)
        k = n - self.f

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


class MoNNA(object):

    r"""
    Description
    -----------

    Apply the MoNNA aggregator [1]_:

    .. math::

        \mathrm{MoNNA}_{f, \mathrm{idx}} \ (x_1, \dots, x_n) = \frac{1}{n-f} \sum_{i \in \mathit{N}_{\mathrm{idx}+1}} x_{i}
        
    where
    
    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - \\(\\mathit{N}_{i}\\) is the set of the \\(n − f\\) nearest neighbors of \\(x_{i}\\) in \\(\\{x_1, \\dots , x_n\\}\\).

    - :math:`\mathrm{idx} \in \{0, \dots, n-1\}` is the ID of the chosen worker/vector for which the neighborhood is computed. In other words, :math:`x_{\mathrm{idx}+1}` is the vector sent by the worker with ID :math:`\mathrm{idx}`.

    Therefore, MoNNA computes the average of the \\(n − f\\) nearest neighbors of the chosen vector with ID :math:`\mathrm{idx}`.
    

    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.
    idx : int, optional
        Index of the vector for which the neighborhood is computed. Set to 0 by default.

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

    MoNNA is used in peer-to-peer settings where :math:`\mathrm{idx}` corresponds to the ID of a vector that is trusted to be correct (i.e., not faulty).

    Examples
    --------
        
    >>> import byzfl
    >>> agg = byzfl.MoNNA(1, 1)

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([2.5, 3.5, 4.5])

    Using torch tensors

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([2.5, 3.5, 4.5])
    
    Using list of torch tensors

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    References
    ----------

    .. [1] Farhadkhani, S., Guerraoui, R., Gupta, N., Hoang, L. N., Pinot, R.,
           & Stephan, J. (2023, July). Robust collaborative learning with 
           linear gradient overhead. In International Conference on Machine 
           Learning (pp. 9761-9813). PMLR. 

    """
    
    def __init__(self, f=0, idx=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f
        if not isinstance(idx, int) or idx < 0:
            raise ValueError("idx must be a non-negative integer")
        self.idx = idx
    
    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")
        if not self.idx < n:
            raise ValueError(f"idx must be smaller than n, but got idx={self.idx} and n={n}")

        distance = distance_tool(vectors)
        dist = distance.cdist(vectors, vectors[self.idx].reshape(1,-1))
        k = n - self.f
        indices = tools.argpartition(dist.reshape(-1), k-1)[:k]
        return tools.mean(vectors[indices], axis=0)


class Meamed(object):

    r"""
    Description
    -----------

    Compute the mean around median along the first axis [1]_:

    .. math::
        \big[\mathrm{Meamed}_{f}(x_1, \ldots, x_n)\big]_k = \frac{1}{n-f} \sum_{j=1}^{n-f} \big[x_{\pi(j)}\big]_k
    
    where 
    
    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.
    
    - \\(\\big[\\cdot\\big]_k\\) refers to the \\(k\\)-th coordinate.

    - :math:`\mathrm{median}` refers to the median of :math:`n` scalars.

    - \\(\\pi\\) denotes a permutation on \\(\\big[n\\big]\\) that sorts the input vectors based on their \\(k\\)-th coordinate in non-decreasing order of distance to the :math:`\mathrm{median}` of the \\(k\\)-th coordinate across the input vectors. This sorting is expressed as:
    
    :math:`\Big|\big[x_{\pi_k(1)}\big]_k - \mathrm{median}\big(\big[x_1\big]_k, \ldots, \big[x_n\big]_k\big)\Big| \leq \ldots \leq \Big|\big[x_{\pi_k(n)}\big]_k - \mathrm{median}\big(\big[x_1\big]_k, \ldots, \big[x_n\big]_k\big)\Big|`.
    
    In other words, Meamed computes the average of the \\(n-f\\) closest elements to the :math:`\mathrm{median}` for each dimension \\(k\\).

    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.

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
        
    >>> import byzfl
    >>> agg = byzfl.Meamed(1)

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([2.5, 3.5, 4.5])

    Using torch tensors

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([2.5, 3.5, 4.5])
    
    Using list of torch tensors

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])


    References
    ----------

    .. [1] Xie, C., Koyejo, O., and Gupta, I. Generalized byzantine-tolerant sgd, 2018.

    """

    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n = len(vectors)
        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")
        
        d = len(vectors[0])
        k = len(vectors) - self.f
        median = tools.median(vectors, axis=0)
        abs_diff = tools.abs((vectors - median))

        indices = tools.argpartition(abs_diff, k-1, axis=0)[:k]
        indices = tools.multiply(indices, d)
        a = tools.arange(d)
        if not tools == np:
            a = a.to(indices.device)
        indices = tools.add(indices, a)
        return tools.mean(vectors.take(indices), axis=0)


class CAF(object):
    r"""
    Description
    -----------

    Implements the **Covariance-bound Agnostic Filter** (CAF) [1]_, a robust aggregator
    designed to tolerate Byzantine inputs without requiring a bound on the covariance
    of honest vectors.

    The algorithm iteratively estimates a robust mean by downweighting samples whose
    deviations from the mean are aligned with the dominant eigenvector of the
    empirical covariance matrix.

    Precisely, given a set of input vectors :math:`x_1, \dots, x_n \in \mathbb{R}^d`,
    the algorithm proceeds as follows:

    1. Initialize weights :math:`c_i = 1` for all :math:`i \in [n]`.
    2. Repeat until the total weight :math:`\sum_i c_i \leq n - 2f`:
        - Compute the weighted empirical mean:

          .. math::
             \mu_c = \frac{1}{\sum_i c_i} \sum_{i=1}^n c_i x_i

        - Using the power method [2]_, compute the dominant eigenvector :math:`v` and maximum eigenvalue :math:`\lambda_{max}` of the empirical covariance matrix:

          .. math::
             \Sigma_c = \frac{1}{\sum_i c_i} \sum_{i=1}^n c_i (x_i - \mu_c)(x_i - \mu_c)^\top

        - For each vector, compute the projection squared:

          .. math::
             \tau_i = ((x_i - \mu_c)^\top v)^2

        - Downweight outliers:

          .. math::
             c_i \leftarrow c_i \cdot \left(1 - \frac{\tau_i}{\max_j \tau_j}\right)

    3. Return the empirical mean :math:`\mu_c` corresponding to the smallest maximum eigenvalue :math:`\lambda_{max}` encountered.

    This algorithm does not assume any upper bound on the spectral norm of the covariance matrix
    and is especially suited to settings with high-dimensional or heterogeneously distributed data.

    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.

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

    >>> import byzfl
    >>> agg = byzfl.CAF(1)

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

    >>> import numpy as np
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

    References
    ----------
    .. [1] Allouah, Y., Guerraoui, R., and Stephan, J. Towards Trustworthy Federated Learning with Untrusted Participants. ICML, 2025.
    .. [2] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.

    """
    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f

    def __call__(self, vectors):
        tools, vectors = check_vectors_type(vectors)
        n, dimension = shape(tools, vectors)

        if self.f * 2 >= n:
            raise ValueError(f"Cannot tolerate 2f ≥ n. Got f={self.f}, n={n}")

        def compute_dominant_eigenvector(tools, vectors, c, diffs, dimension, max_iters=1):
            # Compute the dominant eigenvector using a weighted sum approach.
            if tools == np:
                vector = np.random.randn(dimension)
            else:
                vector = torch.randn(dimension, device=vectors.device)
            vector = vector / tools.linalg.norm(vector)

            eigenvalue = None
            for _ in range(max_iters):
                # Weighted sum for matrix-vector product
                dot_products = tools.matmul(diffs, vector) # (x_i - mu_c) · vector
                weighted_sum = tools.sum(c[:, None] * diffs * dot_products[:, None], axis=0) / tools.sum(c)
                # Normalize the vector
                next_vector = weighted_sum / tools.linalg.norm(weighted_sum)
                # Compute eigenvalue as Rayleigh quotient
                next_eigenvalue = tools.dot(next_vector, weighted_sum)

                vector = next_vector
                eigenvalue = next_eigenvalue

            return eigenvalue, next_vector

        c = ones_vector(tools, n, vectors)
        c_sum = tools.sum(c)
        eigen_val = float('inf')

        best_mu_c = None
        while c_sum > n - 2 * self.f:
            # Compute the empirical mean
            weighted_sum = tools.sum(c[:, None] * vectors, axis=0)
            current_mu_c = weighted_sum / c_sum

            # Compute the maximum eigenvector and eigenvalue of the empirical covariance matrix
            diffs = vectors - current_mu_c      # Compute (x_i - current_mu_c)
            max_eigen_val, max_eigen_vec = compute_dominant_eigenvector(tools, vectors, c, diffs, dimension)

            if max_eigen_val < eigen_val:
                eigen_val = max_eigen_val
                best_mu_c = current_mu_c

            # Compute tau values
            tau = tools.matmul(diffs, max_eigen_vec) ** 2
            # Update weights
            tau_max = tau.max()
            c = c * (1 - tau / tau_max)
            c_sum = tools.sum(c)

        # Return the empirical mean with smallest max_eigen_val encountered
        return best_mu_c


class SMEA(object):
    r"""
    Description
    -----------

    Implements the **Smallest Maximum Eigenvalue Averaging (SMEA)** rule [1]_, a robust aggregation
    method that selects the subset of client vectors whose covariance has the lowest maximum
    eigenvalue, then returns their average.

    Formally, given a set of input vectors :math:`x_1, \dots, x_n \in \mathbb{R}^d` and an integer 
    :math:`f` representing the number of potential Byzantine vectors, the algorithm proceeds as follows:

    1. Enumerate all subsets :math:`S \subset [n]` of size :math:`n - f`.
    2. For each subset :math:`S`, compute its empirical mean:

       .. math::
          \mu_S = \frac{1}{|S|} \sum_{i \in S} x_i

    3. Compute the empirical covariance matrix:

       .. math::
          \Sigma_S = \frac{1}{|S|} \sum_{i \in S} (x_i - \mu_S)(x_i - \mu_S)^\top

    4. Using the power method [2]_, compute the maximum eigenvalue :math:`\lambda_{\max}(\Sigma_S)` of each subset’s covariance.
    5. Select the subset :math:`S^\star` that minimizes the maximum eigenvalue:

       .. math::
          S^\star = \arg\min_{S: |S|=n-f} \lambda_{\max}(\Sigma_S)

    6. Return the empirical mean of the optimal subset :math:`S^\star`:

       .. math::
          \text{SMEA}(x_1, \dots, x_n) = \frac{1}{|S^\star|} \sum_{i \in S^\star} x_i

    While computationally expensive due to its combinatorial nature, SMEA provides state-of-the-art robustness 
    guarantees [1]_. This method is thus particularly well-suited to federated settings where the number of clients is not too large.

    Initialization parameters
    --------------------------
    f : int, optional
        Number of faulty vectors. Set to 0 by default.

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
    >>> import byzfl
    >>> agg = byzfl.SMEA(1)

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([2.5, 3.5, 4.5])

    Using torch tensors

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([2.5, 3.5, 4.5])

    Using list of torch tensors

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([2.5000, 3.5000, 4.5000])

    References
    ----------
    .. [1] Y Allouah, R Guerraoui, N Gupta, R Pinot, J Stephan. On the Privacy-Robustness-Utility Trilemma in Distributed Learning. ICML, 2023.
    .. [2] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.
    """

    def __init__(self, f=0):
        if not isinstance(f, int) or f < 0:
            raise ValueError("f must be a non-negative integer")
        self.f = f

    def __call__(self, vectors):

        tools, vectors = check_vectors_type(vectors)
        n, dimension = shape(tools, vectors)

        if self.f * 2 >= n:
            raise ValueError(f"Too many Byzantine clients (2f >= n). Got f={self.f}, n={n}")

        def compute_dominant_eigenvector(diffs, dimension, max_iters=1):
            if tools == np:
                vector = np.random.randn(dimension)
            else:
                vector = torch.randn(dimension, device=diffs.device)
            vector = vector / tools.linalg.norm(vector)

            eigenvalue = None
            for _ in range(max_iters):
                dot_products = tools.matmul(diffs, vector)
                weighted_sum = tools.sum(diffs * dot_products[:, None], axis=0)
                next_vector = weighted_sum / tools.linalg.norm(weighted_sum)
                next_eigenvalue = tools.dot(next_vector, weighted_sum)

                vector = next_vector
                eigenvalue = next_eigenvalue

            return eigenvalue, next_vector

        def compute_min_subset(vectors, dimension, n, nb_byz):
            min_eigenvalue = float('inf')
            min_subset = None

            for subset in itertools.combinations(range(n), n - nb_byz):
                subset_grads = vectors[tools.asarray(subset)]

                avg = tools.mean(subset_grads, axis=0)
                diffs = subset_grads - avg
                max_eigen_val, _ = compute_dominant_eigenvector(diffs, dimension)

                if max_eigen_val < min_eigenvalue:
                    min_eigenvalue = max_eigen_val
                    min_subset = subset

            return min_subset

        selected_subset = compute_min_subset(vectors, dimension, n, self.f)
        return vectors[tools.asarray(selected_subset)].mean(axis=0)