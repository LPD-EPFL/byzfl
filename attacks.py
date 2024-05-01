import numpy as np

from utils.misc import check_vectors_type

class NoAttack():

    """
    Description
    -----------
    Class representing an attack which behaves like an honest node

    How to use it in experiments
    ----------------------------
    >>> "attack": {
    >>>     "name": "NoAttack",
    >>>     "parameters": {}
    >>> }

    Methods
    ---------
    """
    def get_malicious_vector(self, honest_vectors):
        """
        Compute the arithmetic mean along axis = 0.
        
        Returns the average of the array elements.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Matrix containing arrays whose mean is desired.

        Returns
        -------
        mean_vector : ndarray or torch.tensor
            The mean vector of the input. The dtype of the outputs will be the same as the input.

        Examples
        --------
        With numpy arrays:
            >>> matrix = np.array([[1., 2., 3.], 
            >>>                    [4., 5., 6.], 
            >>>                    [7., 8., 9.]])
            >>> attack = NoAttack()
            >>> result = attack.get_malicious_vector(matrix)
            >>> print(result)
            ndarray([4. 5. 6.])
        With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
            >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
            >>>                       torch.tensor([4., 5., 6.]), 
            >>>                       torch.tensor([7., 8., 9.])])
            >>> attack = NoAttack()
            >>> result = attack.get_malicious_vector(matrix)
            >>> print(result)
            tensor([4., 5., 6.])
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return mean_vector


class SignFlipping:
    """
    Description
    -----------
    Class representing an attack where the sign of the mean vector is flipped.

    How to use it in experiments
    ----------------------------
    >>> "attack": {
    >>>     "name": "SignFlipping",
    >>>     "parameters": {}
    >>> }

    Methods
    -------
    """

    def get_malicious_vector(self, honest_vectors):
        """
        Compute the arithmetic mean along axis = 0 and flip its sign.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Matrix containing arrays whose mean is desired.

        Returns
        -------
        mean_vector : ndarray or torch.tensor
            The mean vector with flipped sign. The dtype of the outputs will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> matrix = np.array([[1., 2., 3.], 
                >>>                    [4., 5., 6.], 
                >>>                    [7., 8., 9.]])
                >>> attack = SignFlipping()
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                ndarray([-4. -5. -6.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
                >>>                       torch.tensor([4., 5., 6.]), 
                >>>                       torch.tensor([7., 8., 9.])])
                >>> attack = SignFlipping()
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                tensor([-4., -5., -6.])
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, -1)

class FallOfEmpires:
    """
    Description
    -----------
    Class representing an attack that scale the mean vector by some factor (1-attack_factor).

    Parameters
    ----------
    attack_factor : int or float
        The factor by which to weaken the mean vector.

    How to use it in experiments
    ----------------------------
    >>> "attack": {
    >>>     "name": "FallOfEmpires",
    >>>     "parameters": {
    >>>         "attack_factor": 3,
    >>>      }
    >>> }

    Methods
    -------
    """

    def __init__(self, attack_factor=3):
        self.attack_factor = attack_factor
    
    def set_attack_parameters(self, factor):
        """
        Set the attack factor for scale the mean vector of honest vectors.

        Parameters
        ----------
        factor : int or float
            The factor by which to weaken the mean vector.
        """
        self.attack_factor = factor

    def get_malicious_vector(self, honest_vectors):
        """
        Compute the weakened arithmetic mean along axis=0.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Matrix containing arrays whose mean is desired.

        Returns
        -------
        mean_vector : ndarray or torch.tensor
            The scaled vector from mean of honest vectors. 
            The dtype of the outputs will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> matrix = np.array([[1., 2., 3.], 
                >>>                    [4., 5., 6.], 
                >>>                    [7., 8., 9.]])
                >>> attack = FallOfEmpires(attack_factor=3)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                ndarray([ -8. -10. -12.])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
                >>>                       torch.tensor([4., 5., 6.]), 
                >>>                       torch.tensor([7., 8., 9.])])
                >>> attack = FallOfEmpires(attack_factor=3)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                tensor([ -8., -10., -12.])
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, 1 - self.attack_factor)


class LittleIsEnough():
    """
    Description
    -----------
    Class representing an attack that perturbs the mean vector of honest vectors by adding a scaled version of the standard deviation vector.
    
    Parameters
    ----------
    attack_factor : int or float
        The factor by which to scale the standard deviation vector.

    How to use it in experiments
    ----------------------------
    >>> "attack": {
    >>>     "name": "LittleIsEnough",
    >>>      "parameters": {
    >>>          "attack_factor": 1.5,
    >>>      }
    >>>  },

    Methods
    -------
    """

    def __init__(self, attack_factor=1.5):
        self.attack_factor = attack_factor
    
    def set_attack_parameters(self, factor):
        """
        Set the attack factor for scaling the standard deviation vector.

        Parameters
        ----------
        factor : (int or float)
            The factor by which to scale the standard deviation vector.
        """
        self.attack_factor = factor

    def get_malicious_vector(self, honest_vectors):
        """
        Perturb the mean vector of honest vectors by adding a scaled version of the standard deviation vector.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Matrix containing arrays of honest vectors.

        Returns
        -------
        malicious_vector : ndarray or torch.tensor
            The perturbed mean vector of honest vectors.
            The dtype of the outputs will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> matrix = np.array([[1., 2., 3.], 
                >>>                    [4., 5., 6.], 
                >>>                    [7., 8., 9.]])
                >>> attack = LittleIsEnough(attack_factor=1.5)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                ndarray([ 8.5  9.5 10.5])
            With torch tensors (Warning: We need the tensor to be either a floating point or complex dtype):
                >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
                >>>                        torch.tensor([4., 5., 6.]), 
                >>>                        torch.tensor([7., 8., 9.])])
                >>> attack = LittleIsEnough(attack_factor=1.5)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                tensor([ 8.5000,  9.5000, 10.5000])
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)
        attack_vector = tools.sqrt(tools.var(honest_vectors, axis=0, ddof=1))
        return tools.add(tools.mean(honest_vectors, axis=0),
                tools.multiply(attack_vector, self.attack_factor))


class Mimic():
    """
    Description
    -----------
    Class representing an attack where the attacker mimics the behavior of a specific worker.

    Parameters
    ----------
    worker_to_mimic : int
            ID of the worker whose behavior is to be mimicked.

    How to use it in experiments
    ----------------------------
    >>> "attack": {
    >>>     "name": "Mimic",
    >>>     "parameters": {
    >>>         "worker_to_mimic": 0
    >>>     }
    >>> }

    Methods
    -------
    """

    def __init__(self, worker_to_mimic=0):
        self.worker_to_mimic = worker_to_mimic
    
    def set_attack_parameters(self, worker):
        """
        Set the worker ID to be mimicked.

        Parameters
        ----------
        worker : int
            ID of the worker whose behavior is to be mimicked.
        """
        self.worker_to_mimic = worker
    
    def get_malicious_vector(self, honest_vectors):
        """
        Retrieve the data from the worker to be mimicked.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor
            Matrix containing arrays of honest workers data.

        Returns
        -------
        malicious_vector : ndarray or torch.tensor
            The data from the worker to be mimicked.
            The dtype of the outputs will be the same as the input.

        Example
        -------
            With numpy arrays:
                >>> matrix = np.array([[1., 2., 3.], 
                                    [4., 5., 6.], 
                                    [7., 8., 9.]])
                >>> attack = Mimic(worker_to_mimic=0)
                >>> attack.set_attack_parameters(0)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                ndarray([1., 2., 3.])
            With torch tensors:
                >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
                                        torch.tensor([4., 5., 6.]), 
                                        torch.tensor([7., 8., 9.])])
                >>> attack = Mimic(worker_to_mimic=0)
                >>> attack.set_attack_parameters(0)
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                tensor([1., 2., 3.])
        """

        return honest_vectors[self.worker_to_mimic]


class Inf():
    """
    Description
    -----------
    Class representing an attack that generates a vector with positive infinity values.

    How to use it in experiments
    ---------------------------
    >>> "attack": {
    >>>     "name": "Inf",
    >>>     "parameters": {}
    >>> }

    Methods
    -------
    """

    def get_malicious_vector(self, honest_vectors):
        """
        Generate a vector with positive infinity values.

        Parameters
        ----------
        honest_vectors : 2D ndarray or 2D torch.tensor
            Matrix containing arrays of honest vectors.

        Returns
        -------
        malicious_vector : ndarray or torch.tensor
            The malicious vector with all elements set to positive infinity.
            The dtype of the outputs will be the same as the input.

        Examples
        --------
            With numpy arrays:
                >>> matrix = np.array([[1., 2., 3.], 
                >>>                    [4., 5., 6.], 
                >>>                    [7., 8., 9.]])
                >>> attack = Inf()
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                ndarray([inf, inf, inf])
            With torch tensors:
                >>> matrix = torch.stack([torch.tensor([1., 2., 3.]),
                >>>                       torch.tensor([4., 5., 6.]), 
                >>>                       torch.tensor([7., 8., 9.])])
                >>> attack = Inf()
                >>> result = attack.get_malicious_vector(matrix)
                >>> print(result)
                tensor([inf, inf, inf])
        """

        tools, honest_vectors = check_vectors_type(honest_vectors)
        return tools.full_like(honest_vectors[0], float('inf'), dtype=np.float64)