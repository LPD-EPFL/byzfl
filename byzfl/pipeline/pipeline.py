import inspect
from byzfl.aggregators import aggregators
from byzfl.aggregators import preaggregators
import byzfl.attacks as attacks

class RobustAggregator:
    """
    Description
    -----------
    The `RobustAggregator` class is a comprehensive utility for applying pre-aggregations and aggregations to a set of input vectors.
    This class combines multiple pre-aggregation steps with a robust aggregation method, ensuring that the input data is processed efficiently and reliably to mitigate the effects of adversarial inputs or outliers.

    Features
    --------
    - **Pre-Aggregation**: Enables the application of multiple pre-aggregation steps in a sequential manner, 
      such as :ref:`clipping-label` or :ref:`nnm-label`, to refine input vectors before aggregation.
    - **Robust Aggregation**: Integrates robust aggregation methods like :ref:`trmean-label` (TrMean) to compute 
      an output vector resilient to Byzantine inputs.
    - **Compatibility**: Works seamlessly with NumPy arrays, PyTorch tensors, and lists of these data types.

    Initialization Parameters
    -------------------------
    aggregator_info : dict 
        A dictionary specifying the aggregation method and its parameters.

        - **Keys**:
            - `"name"`: str
                Name of the aggregation method (e.g., `"TrMean"`).
            - `"parameters"`: dict
                A dictionary of parameters required by the specified aggregation method.
    pre_agg_list : list, optional (default: [])
        A list of dictionaries, each specifying a pre-aggregation method and its parameters.

        - **Keys**:
            - `"name"`: str
                Name of the pre-aggregation method (e.g., `"NNM"`).
            - `"parameters"`: dict
                A dictionary of parameters required by the specified pre-aggregation method.

    Methods
    -------
    aggregate_vectors(vectors)
        Applies the specified pre-aggregation and aggregation methods to the input vectors, returning the aggregated result.

    Calling the Instance
    --------------------
    Input Parameters
    ----------------
    vectors : numpy.ndarray, torch.Tensor, list of numpy.ndarray, or list of torch.Tensor
        A collection of input vectors, matrices, or tensors to process.
        These vectors conceptually correspond to gradients submitted by honest and Byzantine participants during a training iteration.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        The aggregated output vector with the same data type as the input.

    Examples
    --------
    Initialize the `RobustAggregator` with both pre-aggregation and aggregation methods:

    >>> from byzfl import RobustAggregator
    >>> # Define pre-aggregation methods
    >>> pre_aggregators = [
    >>>     {"name": "Clipping", "parameters": {"c": 2.0}},
    >>>     {"name": "NNM", "parameters": {"f": 1}},
    >>> ]
    >>> # Define an aggregation method
    >>> aggregator_info = {"name": "TrMean", "parameters": {"f": 1}}
    >>> # Create the RobustAggregator instance
    >>> rob_agg = RobustAggregator(aggregator_info, pre_agg_list=pre_aggregators)

    Apply the RobustAggregator to various types of input data:

    Using NumPy arrays:

    >>> import numpy as np
    >>> vectors = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> rob_agg.aggregate_vectors(vectors)
    array([0.95841302, 1.14416941, 1.3299258])

    Using PyTorch tensors:

    >>> import torch
    >>> vectors = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> rob_agg.aggregate_vectors(vectors)
    tensor([0.9584, 1.1442, 1.3299])

    Using a list of NumPy arrays:

    >>> import numpy as np
    >>> vectors = [np.array([1., 2., 3.]), np.array([4., 5., 6.]), np.array([7., 8., 9.])]
    >>> rob_agg.aggregate_vectors(vectors)
    array([0.95841302, 1.14416941, 1.3299258])

    Using a list of PyTorch tensors:

    >>> import torch
    >>> vectors = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]), torch.tensor([7., 8., 9.])]
    >>> rob_agg.aggregate_vectors(vectors)
    tensor([0.9584, 1.1442, 1.3299])

    Notes
    -----
    - Pre-aggregations are applied in the order they are listed in `pre_agg_list`.
    - The class dynamically initializes pre-aggregation and aggregation methods based on the provided configurations.
    """

    def __init__(self, aggregator_info, pre_agg_list=[]):
        """
        Initializes the RobustAggregator with the specified pre-aggregation and aggregation configurations.

        Parameters
        ----------
        aggregator_info : dict
            Dictionary specifying the aggregation method and its parameters.
        pre_agg_list : list, optional
            List of dictionaries specifying pre-aggregation methods and their parameters.
        """
        self.aggregator = getattr(aggregators, aggregator_info["name"])
        signature_agg = inspect.signature(self.aggregator.__init__)
        agg_parameters = {
            param.name: aggregator_info["parameters"].get(param.name, param.default)
            for param in signature_agg.parameters.values()
            if param.name in aggregator_info["parameters"]
        }
        self.aggregator = self.aggregator(**agg_parameters)

        self.pre_agg_list = []
        for pre_agg_info in pre_agg_list:
            pre_agg = getattr(preaggregators, pre_agg_info["name"])
            signature_pre_agg = inspect.signature(pre_agg.__init__)
            pre_agg_parameters = {
                param.name: pre_agg_info["parameters"].get(param.name, param.default)
                for param in signature_pre_agg.parameters.values()
                if param.name in pre_agg_info["parameters"]
            }
            self.pre_agg_list.append(pre_agg(**pre_agg_parameters))

    def aggregate_vectors(self, vectors):
        """
        Applies the configured pre-aggregations and robust aggregation method to the input vectors.

        Parameters
        ----------
        vectors : numpy.ndarray, torch.Tensor, list of numpy.ndarray, or list of torch.Tensor
            A collection of input vectors to process.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The aggregated output vector with the same data type as the input.
        """
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        return self.aggregator(vectors)


class ByzantineWorker:
    """
    Description
    -----------
    The `ByzantineWorker` class is responsible for simulating Byzantine behavior in distributed machine learning 
    by executing a specified Byzantine attack. It applies an attack to the gradients (or input vectors) 
    submitted by honest participants and generates multiple faulty (Byzantine) vectors.

    Features
    --------
    - Supports various Byzantine attack strategies through dynamic initialization.
    - Allows customization of attack parameters and the number of faulty nodes.
    - Compatible with both NumPy and PyTorch tensors, as well as lists of these data types.

    Initialization Parameters
    -------------------------
    params : dict
        A dictionary containing the configuration for the Byzantine attack. Must include:

        - `"f"`: int
            The number of faulty (Byzantine) vectors to generate.
        - `"name"`: str
            The name of the attack to be executed (e.g., `"InnerProductManipulation"`).
        - `"parameters"`: dict
            A dictionary of parameters for the specified attack, where keys are parameter names and values are their corresponding values.

    Methods
    -------
    apply_attack(honest_vectors)
        Applies the specified Byzantine attack to the input vectors and returns a list of faulty vectors.

    Calling the Instance
    --------------------
    Input Parameters
    ----------------
    honest_vectors : numpy.ndarray, torch.Tensor, list of numpy.ndarray, or list of torch.Tensor
        A collection of input vectors, matrices, or tensors representing gradients submitted by honest participants.

    Returns
    -------
    list
        A list containing `f` faulty vectors generated by the Byzantine attack, each with the same data type as the input.

    Examples
    --------
    Initialize the `ByzantineWorker` with a specific attack and apply it to input vectors:

    >>> from byzfl import ByzantineWorker
    >>> attack = {
    >>>     "name": "InnerProductManipulation",
    >>>     "f": 3,
    >>>     "parameters": {"tau": 3.0},
    >>> }
    >>> byz_worker = ByzantineWorker(attack)

    Using numpy arrays:

    >>> import numpy as np
    >>> honest_vectors = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> byz_worker.apply_attack(honest_vectors)
    [array([-12., -15., -18.]), array([-12., -15., -18.]), array([-12., -15., -18.])]

    Using torch tensors:

    >>> import torch
    >>> honest_vectors = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> byz_worker.apply_attack(honest_vectors)
    [tensor([-12., -15., -18.]), tensor([-12., -15., -18.]), tensor([-12., -15., -18.])]

    Using a list of numpy arrays:

    >>> import numpy as np
    >>> honest_vectors = [np.array([1., 2., 3.]), np.array([4., 5., 6.]), np.array([7., 8., 9.])]
    >>> byz_worker.apply_attack(honest_vectors)
    [array([-12., -15., -18.]), array([-12., -15., -18.]), array([-12., -15., -18.])]

    Using a list of torch tensors:

    >>> import torch
    >>> honest_vectors = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.]), torch.tensor([7., 8., 9.])]
    >>> byz_worker.apply_attack(honest_vectors)
    [tensor([-12., -15., -18.]), tensor([-12., -15., -18.]), tensor([-12., -15., -18.])]

    """

    def __init__(self, params):
        """
        Initializes the ByzantineWorker with the specified attack configuration.

        Parameters
        ----------
        params : dict
            A dictionary with the attack configuration. Must include:
            - `"f"`: int
                Number of faulty vectors.
            - `"name"`: str
                Name of the attack to execute.
            - `"parameters"`: dict
                Parameters for the specified attack.
        """
        self.f = params["f"]
        self.attack = getattr(
            attacks, 
            params["name"]
        )(**params["parameters"])

    def apply_attack(self, honest_vectors):
        """
        Applies the specified Byzantine attack to the input vectors.

        Parameters
        ----------
        honest_vectors : numpy.ndarray, torch.Tensor, list of numpy.ndarray, or list of torch.Tensor
            A collection of input vectors, matrices, or tensors representing gradients submitted by honest participants.

        Returns
        -------
        list
            A list containing `f` faulty (Byzantine) vectors generated by the attack, each with the same data type as the input.
            If `f = 0`, an empty list is returned.
        """
        if self.f == 0:
            return []
        
        # Generate the Byzantine vector by applying the attack
        byz_vector = self.attack(honest_vectors)

        # Return a list of the same Byzantine vector repeated `f` times
        return [byz_vector] * self.f