import numpy as np
from byzfl.utils.misc import *
from byzfl.aggregators import *

class SignFlipping:
    
    r"""
    Description
    -----------
    
    Send the opposite of the mean vector [1]_.

    .. math::

        \mathrm{SignFlipping} \ (x_1, \dots, x_n) = - \frac{1}{n}\sum_{i=1}^{n} x_i

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.
    
    Initialization parameters
    --------------------------

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
        The data type of the output is the same as the input.

    Examples
    --------

    >>> import byzfl
    >>> attack = byzfl.SignFlipping()

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([-4. -5. -6.])
            
    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([-4., -5., -6.])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([-4., -5., -6.])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([-4., -5., -6.])


    References
    ----------

    .. [1] Zeyuan Allen-Zhu, Faeze Ebrahimianghazani, Jerry Li, and Dan Alistarh. Byzantine-resilient non-convex stochastic gradient descent.
        In International Conference on Learning Representations, 2020

    """

    def __init__(self):
        pass
    
    def __call__(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, -1)


class InnerProductManipulation:
    
    r"""
    Description
    -----------

    Execute the Inner Product Manipulation (IPM) attack [1]_: multiplicatively scale the mean vector by :math:`- \tau`.

    .. math::

        \text{IPM}_{\tau}(x_1, \dots, x_n) = - \tau \cdot \frac{1}{n} \sum_{i=1}^{n} x_i

    where 

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`\tau > 0` is the attack factor.

    Initialization parameters
    --------------------------

    tau : float, optional
        The attack factor :math:`\tau` used to adjust the mean vector. Set to 2.0 by default.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.
    

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.
    
    Examples
    --------

    >>> import byzfl
    >>> attack = byzfl.InnerProductManipulation(2.0)

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([ -8. -10. -12.])
            
    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([-8., -10., -12.])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([ -8. -10. -12.])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([-8., -10., -12.])

    References
    ----------

    .. [1] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Fall of empires: Breaking byzantine-tolerant
            sgd by inner product manipulation. In Ryan P. Adams and Vibhav Gogate (eds.), Proceedings of
            The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine
            Learning Research, pp. 261–270. PMLR, 22–25 Jul 2020. URL https://proceedings.mlr.press/v115/xie20a.html.

    """

    def __init__(self, tau=2.0):
        check_type(tau, float)
        self.tau = tau

    def __call__(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, - self.tau)

class Optimal_InnerProductManipulation:
    
    r"""
    Description
    -----------

    Generalization of the :ref:`ipm-label` attack [1]_ by optimizing the attack factor :math:`\tau`.

    .. math::

        \text{Opt-IPM}_{\textit{agg}, \textit{pre_agg_list}, f}(x_1, \dots, x_n) = - \tau_{opt} \cdot \frac{1}{n} \sum_{i=1}^{n} x_i

    where 

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`\textit{agg}` is the robust aggregator to be used to aggregate the vectors during the training.

    - :math:`\textit{pre_agg_list}` is the list of robust pre-aggregators to be used to transform the vectors during the training.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - :math:`\tau_{opt} \in \mathbb{R}` is the optimal attack factor found using a line-search optimization method.

    This attack is designed to optimize the attack factor :math:`\tau` of the IPM attack by maximizing a specific function.
    The function quantifies the effect of the attack, in particular the \\(\\ell_2\\)-norm of the distance between the aggregated vectors (including Byzantine vectors) and the average of honest vectors.
    The goal is to find the attack factor that results in the maximum disruption.

    This attack, developed by the ByzFL team, draws inspiration from the IPM attack and has been utilized in [2]_.

    Initialization parameters
    --------------------------

    agg : object, optional (default: Average)
        An instance of a robust aggregator that will be used to aggregate the vectors during the optimization process.

    pre_agg_list : list, optional (default: [Clipping])
        A list of pre-aggregation functions, where each element is an object representing a pre-aggregation method.

    f : int, optional (default: 1)
        The number of Byzantine participants. Must be a non-negative integer.

    evals : int, optional (default: 20)
        The maximum number of evaluations during the optimization process. Must be a positive integer.

    start : float, optional (default: 0.0)
        The initial attack factor to evaluate. Must be a float.

    delta : float, optional (default: 10.0)
        The initial step size for the optimization process. Must be a non-zero float.

    ratio : float, optional (default: 0.8)
        The contraction ratio used to reduce the step size during the contraction phase. Must be between 0.5 and 1 (both excluded).

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.

    Examples
    --------

    >>> import byzfl
    >>> # Instantiate the robust aggregator
    >>> agg = byzfl.TrMean(f=1)
    >>> # Instantiate the list of pre-aggregators
    >>> pre_agg_list = [byzfl.NNM(f=1), byzfl.Clipping()]
    >>> # Instantiate the attack
    >>> attack = byzfl.Optimal_InnerProductManipulation(agg, pre_agg_list=pre_agg_list, f=1)

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([-2.98949673 -3.73687091 -4.48424509])
            
    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([-2.9895, -3.7369, -4.4842])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([-2.98949673 -3.73687091 -4.48424509])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([-2.9895, -3.7369, -4.4842])

    References
    ----------

    .. [1] Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Fall of empires: Breaking byzantine-tolerant
            sgd by inner product manipulation. In Ryan P. Adams and Vibhav Gogate (eds.), Proceedings of
            The 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine
            Learning Research, pp. 261–270. PMLR, 22–25 Jul 2020. URL https://proceedings.mlr.press/v115/xie20a.html.
    
    .. [2] Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R.,
           & Stephan, J. (2023, April). Fixing by mixing: A recipe for optimal
           byzantine ml under heterogeneity. In International Conference on 
           Artificial Intelligence and Statistics (pp. 1232-1300). PMLR.  

    """

    def __init__(self, agg=Average(), pre_agg_list=[Clipping()], f=1, evals=20, start=0.0, delta=10.0, ratio=0.8):

        # List of valid aggregator classes
        valid_agg_classes = (Average, Median, TrMean, GeometricMedian, Krum, MultiKrum, CenteredClipping, MDA, MoNNA, Meamed)
        check_type(agg, valid_agg_classes)
        self.agg = agg

        # List of valid pre-aggregator classes
        check_type(pre_agg_list, list)
        valid_pre_agg_classes = (NNM, Bucketing, Clipping, ARC)
        for pre_agg in pre_agg_list:
            check_type(pre_agg, valid_pre_agg_classes)
        self.pre_agg_list = pre_agg_list

        check_type(f, int)
        check_greater_than_or_equal_value(f, "f", 0)
        self.f = f
        check_type(evals, int)
        check_greater_than_value(evals, "evals", 0)
        self.evals = evals
        check_type(start, float)
        self.start = start
        check_type(delta, float)
        check_different_from_value(delta, "delta", 0.0)
        self.delta = delta
        check_type(ratio, float)
        check_greater_than_value(ratio, "ratio", 0.5)
        check_smaller_than_value(ratio, "ratio", 1.0)
        
        self.ratio = ratio

    def _evaluate(self, honest_vectors, avg_honest_vector, current_tau):
       
        """
        Computes the norm of the distance between the aggregated vector (including Byzantine vectors) and the average of honest vectors.

        Parameters
        ----------
        honest_vectors : ndarray or torch.tensor
            Honest node vectors (2D).
        avg_honest_vector : ndarray or torch.tensor
            Average of honest_vectors (1D).
        current_tau : float
            Current attack factor considered in the attack.

        Returns
        -------
        float
            Norm of the distance.
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)

        #Compute Byzantine vector
        attack = InnerProductManipulation(tau=current_tau)
        byzantine_vector = attack(honest_vectors)
        byzantine_vectors = tools.array([byzantine_vector] * self.f)

        #Aggregate vectors with current Byzantine vectors
        vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        aggregated_vector = self.agg(vectors)

        #Return distance between aggregate vector and mean of honest vectors
        distance = tools.subtract(aggregated_vector, avg_honest_vector)
        return tools.linalg.norm(distance).item()

    def _expansion_phase(self, honest_vectors, avg_honest_vector):
        
        """
        Performs the expansion phase of the optimization.
        This phase explores the search space aggressively by increasing the step size (delta) when a better value is found.
        """
        best_x = self.start
        best_y = self._evaluate(honest_vectors, avg_honest_vector, best_x)
        delta = self.delta
        remaining_evals = self.evals - 1

        while remaining_evals > 0:
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If the new value is better: Update best_x, double the step size (delta *= 2), and continue exploring.
                best_x, best_y = prop_x, prop_y
                delta *= 2
            else:
                # If the new value is worse: Stop the expansion phase and proceed to contraction.
                delta *= self.ratio
                break

        return best_x, best_y, delta, remaining_evals

    def _contraction_phase(self, honest_vectors, avg_honest_vector, best_x, best_y, delta, remaining_evals):

        """
        Performs the contraction phase of the optimization.
        This phase refines the search by reducing the step size (delta) and searching around the current best value.
        """

        while remaining_evals > 0:
            # Continue reducing the step size until no significant improvements are found or evaluations are exhausted.
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If a better value is found: Update best_x and best_y.
                best_x, best_y = prop_x, prop_y

            delta *= self.ratio

        return best_x

    def __call__(self, honest_vectors):

        """
        Iteratively computes the best attack factor tau and executes the corresponding IPM attack.

        Parameters
        ----------
        honest_vectors : ndarray or torch.tensor
            Honest node vectors (2D).
        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        avg_honest_vector = tools.mean(honest_vectors, axis=0)

        # Expansion Phase
        best_tau, largest_distance, delta, remaining_evals = self._expansion_phase(honest_vectors, avg_honest_vector)

        # Contraction Phase
        best_tau = self._contraction_phase(honest_vectors, avg_honest_vector, best_tau, largest_distance, delta, remaining_evals)

        # Set the best attack factor and execute IPM
        attack = InnerProductManipulation(tau=best_tau)
        return attack(honest_vectors)


class ALittleIsEnough:

    r"""
    Description
    -----------

    Execute the A Little Is Enough (ALIE) attack [1]_: perturb the mean vector using the coordinate-wise standard deviation of the vectors multiplicatively scaled with the attack factor :math:`\tau`.

    .. math::

        \text{ALIE}_{\tau}(x_1, \dots, x_n) = \mu_{x_1, ..., x_n} + \tau \cdot \sigma_{x_1, ..., x_n}
    
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`\mu_{x_1, \dots, x_n} = \frac{1}{n}\sum_{i=1}^{n}x_i` is the mean vector.

    - \\(\\big[\\cdot\\big]_k\\) refers to the \\(k\\)-th coordinate.

    - :math:`\sigma_{x_1, \dots, x_n}` is the coordinate-wise standard deviation of :math:`x_1, \dots, x_n`, i.e., :math:`\big[\sigma_{x_1, \dots, x_n}\big]_k = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\big[x_i\big]_k - \big[\mu_{x_1, \dots, x_n}\big]_k)^2}`.

    - :math:`\tau \in \mathbb{R}` is the attack factor.

    Initialization parameters
    --------------------------

    tau : float, optional
        The attack factor :math:`\tau` used to adjust the mean vector. Set to 1.5 by default.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.
    
    Examples
    --------

    >>> import byzfl
    >>> attack = byzfl.ALittleIsEnough(1.5)

    Using numpy arrays:

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([ 8.5  9.5 10.5])
    
    Using torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
    
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([ 8.5000,  9.5000, 10.5000])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([ 8.5  9.5 10.5])

    Using list of torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([ 8.5000,  9.5000, 10.5000])
   
    References
    ----------

    .. [1] Baruch, M., Baruch, G., and Goldberg, Y. A little is enough: Circumventing defenses for distributed learning.
           In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, 8-14 December 2019, Long Beach, CA, USA, 2019.

    """

    def __init__(self, tau=1.5):
        check_type(tau, float)
        self.tau = tau

    def __call__(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        attack_vector = tools.sqrt(tools.var(honest_vectors, axis=0, ddof=1))
        return tools.add(tools.mean(honest_vectors, axis=0), tools.multiply(attack_vector, self.tau))

class Optimal_ALittleIsEnough:
    
    r"""
    Description
    -----------

    Generalization of the :ref:`alie-label` attack [1]_ by optimizing the attack factor :math:`\tau`. 

    .. math::

        \text{Opt-ALIE}_{\textit{agg}, \textit{pre_agg_list}, f}(x_1, \dots, x_n) = \mu_{x_1, ..., x_n} + \tau_{opt} \cdot \sigma_{x_1, ..., x_n}

    where 

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`\textit{agg}` is the robust aggregator to be used to aggregate the vectors during the training.

    - :math:`\textit{pre_agg_list}` is the list of robust pre-aggregators to be used to transform the vectors during the training.

    - :math:`f` conceptually represents the expected number of Byzantine vectors.

    - :math:`\tau_{opt} \in \mathbb{R}` is the optimal attack factor found using a line-search optimization method.

    - :math:`\mu_{x_1, \dots, x_n} = \frac{1}{n}\sum_{i=1}^{n}x_i` is the mean vector.

    - \\(\\big[\\cdot\\big]_k\\) refers to the \\(k\\)-th coordinate.

    - :math:`\sigma_{x_1, \dots, x_n}` is the coordinate-wise standard deviation of :math:`x_1, \dots, x_n`, i.e., :math:`\big[\sigma_{x_1, \dots, x_n}\big]_k = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(\big[x_i\big]_k - \big[\mu_{x_1, \dots, x_n}\big]_k)^2}`.

    This attack is designed to optimize the attack factor :math:`\tau` of the ALIE attack by maximizing a specific function.
    The function quantifies the effect of the attack, in particular the \\(\\ell_2\\)-norm of the distance between the aggregated vectors (including Byzantine vectors) and the average of honest vectors.
    The goal is to find the attack factor that results in the maximum disruption.

    This attack, developed by the ByzFL team, draws inspiration from the ALIE attack and has been utilized in [2]_.

    Initialization parameters
    --------------------------

    agg : object, optional (default: Average)
        An instance of a robust aggregator that will be used to aggregate the vectors during the optimization process.

    pre_agg_list : list, optional (default: [Clipping])
        A list of pre-aggregation functions, where each element is an object representing a pre-aggregation method.

    f : int, optional (default: 1)
        The number of Byzantine participants. Must be a non-negative integer.

    evals : int, optional (default: 20)
        The maximum number of evaluations during the optimization process. Must be a positive integer.

    start : float, optional (default: 0.0)
        The initial attack factor to evaluate. Must be a float.

    delta : float, optional (default: 10.0)
        The initial step size for the optimization process. Must be a non-zero float.

    ratio : float, optional (default: 0.8)
        The contraction ratio used to reduce the step size during the contraction phase. Must be between 0.5 and 1 (both excluded).

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.

    Examples
    --------

    >>> import byzfl
    >>> # Instantiate the robust aggregator
    >>> agg = byzfl.TrMean(f=1)
    >>> # Instantiate the list of pre-aggregators
    >>> pre_agg_list = [byzfl.NNM(f=1), byzfl.Clipping()]
    >>> # Instantiate the attack
    >>> attack = byzfl.Optimal_ALittleIsEnough(agg, pre_agg_list=pre_agg_list, f=1)

    Using numpy arrays
        
    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([6.0615843 7.0615843 8.0615843])
            
    Using torch tensors
        
    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([6.0616, 7.0616, 8.0616])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([6.0615843 7.0615843 8.0615843])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([6.0616, 7.0616, 8.0616])

    References
    ----------

    .. [1] Baruch, M., Baruch, G., and Goldberg, Y. A little is enough: Circumventing defenses for distributed learning.
           In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, 8-14 December 2019, Long Beach, CA, USA, 2019.
    
    .. [2] Allouah, Y., Farhadkhani, S., Guerraoui, R., Gupta, N., Pinot, R.,
           & Stephan, J. (2023, April). Fixing by mixing: A recipe for optimal
           byzantine ml under heterogeneity. In International Conference on 
           Artificial Intelligence and Statistics (pp. 1232-1300). PMLR.  

    """

    def __init__(self, agg=Average(), pre_agg_list=[Clipping()], f=1, evals=20, start=0.0, delta=10.0, ratio=0.8):

        # List of valid aggregator classes
        valid_agg_classes = (Average, Median, TrMean, GeometricMedian, Krum, MultiKrum, CenteredClipping, MDA, MoNNA, Meamed)
        check_type(agg, valid_agg_classes)
        self.agg = agg

        # List of valid pre-aggregator classes
        check_type(pre_agg_list, list)
        valid_pre_agg_classes = (NNM, Bucketing, Clipping, ARC)
        for pre_agg in pre_agg_list:
            check_type(pre_agg, valid_pre_agg_classes)
        self.pre_agg_list = pre_agg_list

        check_type(f, int)
        check_greater_than_or_equal_value(f, "f", 0)
        self.f = f
        check_type(evals, int)
        check_greater_than_value(evals, "evals", 0)
        self.evals = evals
        check_type(start, float)
        self.start = start
        check_type(delta, float)
        check_different_from_value(delta, "delta", 0.0)
        self.delta = delta
        check_type(ratio, float)
        check_greater_than_value(ratio, "ratio", 0.5)
        check_smaller_than_value(ratio, "ratio", 1.0)
        
        self.ratio = ratio

    def _evaluate(self, honest_vectors, avg_honest_vector, current_tau):
       
        """
        Computes the norm of the distance between the aggregated vector (including Byzantine vectors) and the average of honest vectors.

        Parameters
        ----------
        honest_vectors : ndarray or torch.tensor
            Honest node vectors (2D).
        avg_honest_vector : ndarray or torch.tensor
            Average of honest_vectors (1D).
        current_tau : float
            Current attack factor considered in the attack.

        Returns
        -------
        float
            Norm of the distance.
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)

        #Compute Byzantine vector
        attack = ALittleIsEnough(tau=current_tau)
        byzantine_vector = attack(honest_vectors)
        byzantine_vectors = tools.array([byzantine_vector] * self.f)

        #Aggregate vectors with current Byzantine vectors
        vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        aggregated_vector = self.agg(vectors)

        #Return distance between aggregate vector and mean of honest vectors
        distance = tools.subtract(aggregated_vector, avg_honest_vector)
        return tools.linalg.norm(distance).item()

    def _expansion_phase(self, honest_vectors, avg_honest_vector):
        
        """
        Performs the expansion phase of the optimization.
        This phase explores the search space aggressively by increasing the step size (delta) when a better value is found.
        """
        best_x = self.start
        best_y = self._evaluate(honest_vectors, avg_honest_vector, best_x)
        delta = self.delta
        remaining_evals = self.evals - 1

        while remaining_evals > 0:
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If the new value is better: Update best_x, double the step size (delta *= 2), and continue exploring.
                best_x, best_y = prop_x, prop_y
                delta *= 2
            else:
                # If the new value is worse: Stop the expansion phase and proceed to contraction.
                delta *= self.ratio
                break

        return best_x, best_y, delta, remaining_evals

    def _contraction_phase(self, honest_vectors, avg_honest_vector, best_x, best_y, delta, remaining_evals):

        """
        Performs the contraction phase of the optimization.
        This phase refines the search by reducing the step size (delta) and searching around the current best value.
        """
        while remaining_evals > 0:
            # Continue reducing the step size until no significant improvements are found or evaluations are exhausted.
            prop_x = best_x + delta
            prop_y = self._evaluate(honest_vectors, avg_honest_vector, prop_x)
            remaining_evals -= 1

            if prop_y > best_y:
                # If a better value is found: Update best_x and best_y.
                best_x, best_y = prop_x, prop_y

            delta *= self.ratio

        return best_x

    def __call__(self, honest_vectors):

        """
        Iteratively computes the best attack factor tau and executes the corresponding IPM attack.

        Parameters
        ----------
        honest_vectors : ndarray or torch.tensor
            Honest node vectors (2D).
        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        avg_honest_vector = tools.mean(honest_vectors, axis=0)

        # Expansion Phase
        best_tau, largest_distance, delta, remaining_evals = self._expansion_phase(honest_vectors, avg_honest_vector)

        # Contraction Phase
        best_tau = self._contraction_phase(honest_vectors, avg_honest_vector, best_tau, largest_distance, delta, remaining_evals)

        # Set the best attack factor and execute IPM
        attack = ALittleIsEnough(tau=best_tau)
        return attack(honest_vectors)


class Mimic:
    
    r"""
    Description
    -----------

    The attacker mimics the behavior of worker with ID :math:`\epsilon` by sending the same vector as that worker [1]_.

    .. math::

        \text{Mimic}_{\epsilon}(x_1, \dots, x_n) = x_{\epsilon+1}
    
    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`\epsilon \in \{0, \dots, n-1\}` is the ID of the worker to mimic. In other words, :math:`x_{\epsilon+1}` is the vector sent by the worker with ID :math:`\epsilon`.

    Initialization parameters
    --------------------------

    epsilon : int, optional
        ID of the worker whose behavior is to be mimicked. Set to 0 by default.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.
    
    Examples
    --------
    
    >>> import byzfl
    >>> attack = byzfl.Mimic(0)

    Using numpy arrays:

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([1. 2. 3.])

    Using torch tensors:

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([1., 2., 3.])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([1.  2. 3.])

    Using list of torch tensors:

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([1., 2., 3.])

    References
    ----------

    .. [1] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Byzantine-robust learning on heterogeneous datasets via bucketing. In International Conference on Learning Representations, 2022.
    
    """

    def __init__(self, epsilon=0):
        check_type(epsilon, int)
        check_greater_than_or_equal_value(epsilon, "epsilon", 0)
        self.epsilon = epsilon

    def __call__(self, honest_vectors):
        check_smaller_than_value(self.epsilon, "epsilon", len(honest_vectors))
        return honest_vectors[self.epsilon]


class Inf:

    r"""
    Description
    -----------

    Generate extreme vector comprised of positive infinity values.

    .. math::

        \mathrm{Inf}(x_1, \dots, x_n) = \begin{bmatrix} +\infty \\ +\infty \\ \vdots \\ +\infty \end{bmatrix} \in \mathbb{R}^d

    where

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`d` is the dimensionality of the input space, i.e., :math:`d` is the number of coordinates of vectors :math:`x_1, \dots, x_n`.

    Initialization parameters
    --------------------------

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
        The data type of the output is the same as the input.

    Examples
    --------

    >>> import byzfl
    >>> attack = byzfl.Inf()

    Using numpy arrays:

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([inf, inf, inf])
    
    Using torch tensors:

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([inf, inf, inf])

    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([inf, inf, inf])

    Using list of torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([inf, inf, inf])

    """ 

    def __init__(self):
        pass
    
    def __call__(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        return tools.full_like(honest_vectors[0], float('inf'), dtype=np.float64)


class Gaussian:

    r"""
    Description
    -----------

    Generate a random vector where each coordinate is independently sampled from a Gaussian distribution.

    .. math::

        \mathrm{Gaussian}_{\mu, \sigma}(x_1, \dots, x_n) = \begin{bmatrix} g_1 \\ g_2 \\ \vdots \\ g_d \end{bmatrix} \in \mathbb{R}^d
        
    where:

    - :math:`x_1, \dots, x_n` are the input vectors, which conceptually correspond to correct gradients submitted by honest participants during a training iteration.

    - :math:`d` is the dimensionality of the input space, i.e., :math:`d` is the number of coordinates of vectors :math:`x_1, \dots, x_n`.

    - :math:`\mathit{N}(\mu, \sigma^2)` is the Gaussian distribution of mean :math:`\mu \in \mathbb{R}` and standard deviation :math:`\sigma \geq 0`.

    - :math:`g_i  \sim \mathit{N}(\mu, \sigma^2)` for all :math:`i  \in \{1, \dots, d\}`.


    Initialization parameters
    --------------------------

    mu: float, optional (default=0.0)
        Mean of the Gaussian distribution.
    sigma: float, optional (default=1.0)
        Standard deviation of the Gaussian distribution.

    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output is the same as the input.

    Examples
    --------

    >>> import byzfl
    >>> attack = byzfl.Gaussian(mu=0.0, sigma=1.0)

    Using numpy arrays

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> attack(x)
    array([-0.08982162  0.07237574  0.55886579])
            
    Using torch tensors

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> attack(x)
    tensor([ 0.9791,  0.0266, -1.0112])
    
    Using list of numpy arrays

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray  
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> attack(x)
    array([-0.08982162  0.07237574  0.55886579])

    Using list of torch tensors
        
    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> attack(x)
    tensor([ 0.9791,  0.0266, -1.0112])

    """

    def __init__(self, mu=0.0, sigma=1.0):
        check_type(mu, float)
        self.mu = mu
        check_type(sigma, float)
        check_greater_than_or_equal_value(sigma, "sigma", 0)
        self.sigma = sigma

    def __call__(self, honest_vectors):
        _, honest_vectors = check_vectors_type(honest_vectors)
        random = random_tool(honest_vectors)
        shape = honest_vectors.shape[1]
        return random.normal(loc=self.mu, scale=self.sigma, size=shape)