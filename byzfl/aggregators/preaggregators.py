import math
import byzfl.utils.misc as misc

class NNM(object):

    r"""
    Description
    -----------

    Apply the Nearest Neighbor Mixing pre-aggregation rule [1]_:

    .. math::

        \mathrm{NNM}_{f} \ (x_1, \dots, x_n) = \left(\frac{1}{n-f}\sum_{i \in \mathit{N}_{1}} x_i \ \ , \ \dots \ ,\ \  \frac{1}{n-f}\sum_{i \in \mathit{N}_{n}} x_i \right)
        
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - For any \\(i \\in \\) \\(\\big[n\\big]\\), \\(\\mathit{N}_i\\) is the set of the \\(n-f\\) nearest neighbors of \\(x_i\\) in \\(\\{x_1, \\dots , x_n\\}\\).

    
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
    >>> agg = byzfl.NNM(1)

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

    >>> import numpy as np
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

    def __init__(self, f=0):
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors))

        distance = misc.distance_tool(vectors)
        dist = distance.cdist(vectors, vectors)
        k = len(vectors) - self.f
        indices = tools.argpartition(dist, k-1, axis = 1)[:,:k]
        return tools.mean(vectors[indices], axis = 1)


class Bucketing(object):

    r"""
    Description
    -----------

    Apply the Bucketing pre-aggregation rule [1]_:

    .. math::

        \mathrm{Bucketing}_{s} \ (x_1, \dots, x_n) = 
        \left(\frac{1}{s}\sum_{i=1}^s x_{\pi(i)} \ \ , \ \ 
        \frac{1}{s}\sum_{i=s+1}^{2s} x_{\pi(i)} \ \ , \ \dots \ ,\ \  
        \frac{1}{s}\sum_{i=\left(\lceil n/s \rceil-1\right)s+1}^{n} x_{\pi(i)} \right)

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - \\(\\pi\\) is a random permutation on \\(\\big[n\\big]\\).

    - \\(s > 0\\) is the bucket size, i.e., the number of vectors per bucket.

    Initialization parameters
    --------------------------
    s : int, optional
        Number of vectors per bucket. Set to 1 by default.
    
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
    >>> agg = byzfl.Bucketing(2)

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

    >>> import numpy as np
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

    def __init__(self, s=1):
        misc.check_type(s, int)
        misc.check_greater_than_value(s, "s", 0)
        self.s = s

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
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


class Identity(object):

    def __init__(self):
        pass

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        return tools.copy(vectors)


class Clipping(object):

    r"""
    Description
    -----------

    Apply the Static Clipping pre-aggregation rule:

    .. math::

        \mathrm{Clipping}_{c} \ (x_1, \dots, x_n) = 
        \left( \min\left\{1, \frac{c}{\big|\big|x_1\big|\big|_2}\right\} x_1 \ \ , \ \dots \ ,\ \  
        \min\left\{1, \frac{c}{\big|\big|x_n\big|\big|_2}\right\} x_n \right)

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - \\(c \\geq 0\\) is the static clipping threshold. Any input vector with an \\(\\ell_2\\)-norm greater than \\(c\\) will be will be scaled down such that its \\(\\ell_2\\)-norm equals \\(c\\).

    Initialization parameters
    --------------------------
    c : float, optional
        Static clipping threshold. Set to 2.0 by default.
    
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
    >>> agg = byzfl.Clipping(2.0)

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([[0.53452248, 1.06904497, 1.60356745],
            [0.91168461, 1.13960576, 1.36752692],
            [1.00514142, 1.14873305, 1.29232469]])

    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([[0.5345, 1.0690, 1.6036],
            [0.9117, 1.1396, 1.3675],
            [1.0051, 1.1487, 1.2923]])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([[0.53452248, 1.06904497, 1.60356745],
            [0.91168461, 1.13960576, 1.36752692],
            [1.00514142, 1.14873305, 1.29232469]])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of  torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([[0.5345, 1.0690, 1.6036],
            [0.9117, 1.1396, 1.3675],
            [1.0051, 1.1487, 1.2923]])
 
    """

    def __init__(self, c=2.0):
        misc.check_type(c, float)
        misc.check_greater_than_or_equal_value(c, "c", 0.0)
        self.c = c

    def _clip_vector(self, vector):
        tools, vector = misc.check_vectors_type(vector)
        vector_norm = tools.linalg.norm(vector)
        if vector_norm > self.c:
            vector = tools.multiply(vector, self.c / vector_norm)
        return vector

    def __call__(self, vectors):
        for i in range(len(vectors)):
            vectors[i] = self._clip_vector(vectors[i])
        return vectors


class ARC(object):

    r"""
    Description
    -----------

    Apply the Adaptive Robust Clipping pre-aggregation rule [1]_:

    .. math::

        \mathrm{ARC}_{f} \ (x_1, \dots, x_n) = 
        \left( \min\left\{1, \frac{x_{\pi(k)}}{\big|\big|x_1\big|\big|_2}\right\} x_1 \ \ , \ \dots \ ,\ \  
        \min\left\{1, \frac{x_{\pi(k)}}{\big|\big|x_n\big|\big|_2}\right\} x_n \right)

    where 

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - :math:`\big|\big|.\big|\big|_2` denotes the \\(\\ell_2\\)-norm.

    - \\(k = \\lfloor 2 \\cdot \\frac{f}{n} \\cdot (n - f) \\rfloor\\).
    
    - \\(\\pi\\) denotes a permutation on \\(\\big[n\\big]\\) that sorts the \\(\\ell_2\\)-norm of the input vectors in non-increasing order. This sorting is expressed as: :math:`\big|\big|x_{\pi(1)}\big|\big|_2 \leq \ldots \leq \big|\big|x_{\pi(n)}\big|\big|_2`.

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
    >>> agg = byzfl.ARC(1)

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> agg(x)
    array([[1.        , 2.        , 3.        ],
            [4.        , 5.        , 6.        ],
            [4.41004009, 5.04004582, 5.67005155]])

    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> agg(x)
    tensor([[1.0000, 2.0000, 3.0000],
            [4.0000, 5.0000, 6.0000],
            [4.4100, 5.0400, 5.6701]])
    
    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> agg(x)
    array([[1.        , 2.        , 3.        ],
            [4.        , 5.        , 6.        ],
            [4.41004009, 5.04004582, 5.67005155]])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of  torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> agg(x)
    tensor([[1.0000, 2.0000, 3.0000],
            [4.0000, 5.0000, 6.0000],
            [4.4100, 5.0400, 5.6701]])

    References
    ----------

    .. [1] Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, and John Stephan. 2024.
           The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning. arXiv:2405.14432 [cs.LG] https://arxiv.org/abs/2405.14432.
 
    """

    def __init__(self, f=0):
        misc.check_type(f, int)
        misc.check_greater_than_or_equal_value(f, "f", 0)
        self.f = f

    def _clip_vector(self, vector, clip_threshold):
        tools, vector = misc.check_vectors_type(vector)
        vector_norm = tools.linalg.norm(vector)
        if vector_norm > clip_threshold:
            vector = tools.multiply(vector, (clip_threshold / vector_norm))
        return vector

    def __call__(self, vectors):
        tools, vectors = misc.check_vectors_type(vectors)
        misc.check_smaller_than_value(self.f, "f", len(vectors)+1)
        magnitudes = [(tools.linalg.norm(vector), vector_id) for vector_id, vector in enumerate(vectors)]
        magnitudes.sort(key=lambda x:x[0])
        nb_vectors = len(vectors)
        nb_clipped = int((2 * self.f / nb_vectors) * (nb_vectors - self.f))
        cut_off_value = nb_vectors - nb_clipped
        f_largest = magnitudes[cut_off_value:]
        clipping_threshold = magnitudes[cut_off_value - 1][0]
        for _, vector_id in f_largest:
            vectors[vector_id] = self._clip_vector(vectors[vector_id], clipping_threshold)
        return vectors