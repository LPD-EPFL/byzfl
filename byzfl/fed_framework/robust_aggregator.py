from byzfl.aggregators import aggregators, preaggregators
import inspect

class RobustAggregator:
    """
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

        # Check for correct types and values in params
        if not isinstance(aggregator_info, dict):
            raise TypeError(f"'aggregator_info' must be of type dict, but got {type(aggregator_info).__name__}")
        if not isinstance(pre_agg_list, list) or not all(isinstance(pre_agg, dict) for pre_agg in pre_agg_list):
            raise TypeError(f"'pre_agg_list' must be must be a list of dict")

        # Initialize the RobustAggregator instance
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