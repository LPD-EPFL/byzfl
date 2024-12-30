import itertools
import numpy as np
import torch
from byzfl.utils import misc


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
        tools, vectors = misc.check_vectors_type(vectors)
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
        tools, vectors = misc.check_vectors_type(vectors)
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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors)/2)
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
        misc.check_type(nu, float)
        self.nu = nu
        misc.check_type(T, int)
        misc.check_greater_than_or_equal_value(T, "T", 0)
        self.T = T

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f
    
    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors)-1)

        distance = misc.distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis=1)[:,1:len(vectors)-self.f]
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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors)-1)

        distance = misc.distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)**2
        dist = tools.sort(dist, axis = 1)[:,1:len(vectors)-self.f]
        dist = tools.mean(dist, axis = 1)
        k = len(vectors) - self.f
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
        if m is not None:
            misc.check_type(self.m, (np.ndarray, torch.Tensor))
        self.m = m
        misc.check_type(L, int)
        misc.check_greater_than_or_equal_value(L, "L", 0)
        self.L = L
        misc.check_type(tau, float)
        misc.check_greater_than_or_equal_value(tau, "tau", 0.0)
        self.tau = tau

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)

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

    Apply the Minimum-Diameter Averaging aggregator [1]_:

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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f
    
    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors))

        distance = misc.distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)
        n = len(vectors)
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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f
        misc.check_type(idx, int)
        misc.check_greater_than_or_equal_value(idx, "idx", 0)
        self.idx = idx
    
    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors))
        misc.check_smaller_than_value(self.idx, "idx", len(vectors))

        distance = misc.distance_tool(vectors)
        dist = distance.cdist(vectors, vectors[self.idx].reshape(1,-1))
        k = len(vectors) - self.f
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
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors))
        
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