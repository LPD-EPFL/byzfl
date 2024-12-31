import inspect
from byzfl.aggregators import aggregators
from byzfl.aggregators import preaggregators

class RobustAggregator:

    """
    Description
    -----------
    The `RobustAggregator` class is a comprehensive utility for applying pre-aggregations and aggregations to a set of input vectors. This class combines multiple pre-aggregation steps with a robust aggregation method, ensuring that the input data is processed efficiently and reliably to mitigate the effects of adversarial inputs or outliers.

    Features
    --------
    - **Pre-Aggregation**: Allows the application of multiple pre-aggregation techniques sequentially, such as :ref:`clipping-label` or :ref:`nnm-label`, to refine input vectors.
    - **Robust Aggregation**: Integrates robust aggregation methods like :ref:`trmean-label` (TrMean) to compute an output vector resilient to Byzantine inputs.

    Initialization Parameters
    -------------------------
    aggregator_info : dict 
        Specifies the aggregation method and its parameters.
        - **Keys**:
            - `"name"`: Name of the aggregation method (e.g., `"TrMean"`).
            - `"parameters"`: Dictionary of parameters required by the specified aggregation method.
    pre_agg_list : list (default: [])
        A list of dictionaries, each specifying a pre-aggregation method and its parameters.
        - **Keys**:
            - `"name"`: Name of the pre-aggregation method (e.g., `"NNM"`).
            - `"parameters"`: Dictionary of parameters required by the specified pre-aggregation method.

    Calling the Instance
    --------------------
    Input Parameters
    ----------------
    vectors : numpy.ndarray, torch.Tensor, list of numpy.ndarray, or list of torch.Tensor
        A collection of input vectors, matrices, or tensors to process.

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

    Using numpy arrays:

    >>> import numpy as np
    >>> x = np.array([[1., 2., 3.],       # np.ndarray
    >>>               [4., 5., 6.], 
    >>>               [7., 8., 9.]])
    >>> rob_agg(x)
    array([0.95841302, 1.14416941, 1.3299258])

    Using torch tensors:

    >>> import torch
    >>> x = torch.tensor([[1., 2., 3.],   # torch.tensor 
    >>>                   [4., 5., 6.], 
    >>>                   [7., 8., 9.]])
    >>> rob_agg(x)
    tensor([0.9584, 1.1442, 1.3299])

    Using a list of numpy arrays:

    >>> import numpy as np
    >>> x = [np.array([1., 2., 3.]),      # list of np.ndarray 
    >>>      np.array([4., 5., 6.]), 
    >>>      np.array([7., 8., 9.])]
    >>> rob_agg(x)
    array([0.95841302, 1.14416941, 1.3299258])

    Using a list of torch tensors:

    >>> import torch
    >>> x = [torch.tensor([1., 2., 3.]),  # list of torch.tensor 
    >>>      torch.tensor([4., 5., 6.]), 
    >>>      torch.tensor([7., 8., 9.])]
    >>> rob_agg(x)
    tensor([0.9584, 1.1442, 1.3299])

    Notes
    -----
    - The class dynamically initializes pre-aggregation and aggregation methods based on the provided configurations.
    - Pre-aggregations are applied in the order they are listed in `pre_agg_list`.

    """

    def __init__(self, aggregator_info, pre_agg_list=[]):

        self.aggregator = getattr(aggregators, aggregator_info["name"])
        signature_agg = inspect.signature(self.aggregator.__init__)
        agg_parameters = {}
        for parameter in signature_agg.parameters.values():
            param_name = parameter.name
            if param_name in aggregator_info["parameters"]:
                agg_parameters[param_name] = aggregator_info["parameters"][param_name]
        self.aggregator = self.aggregator(**agg_parameters)

        self.pre_agg_list = []
        for pre_agg_info in pre_agg_list:
            pre_agg = getattr(preaggregators, pre_agg_info["name"])
            signature_pre_agg = inspect.signature(pre_agg.__init__)
            pre_agg_parameters = {}
            for parameter in signature_pre_agg.parameters.values():
                param_name = parameter.name
                if param_name in pre_agg_info["parameters"]:
                    pre_agg_parameters[param_name] = pre_agg_info["parameters"][param_name]
            pre_agg = pre_agg(**pre_agg_parameters)
            self.pre_agg_list.append(pre_agg)

    def __call__(self, vectors):
        """
        Description
        -----------
        Apply pre-aggregations and aggregations to the vectors
        """

        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        return self.aggregator(vectors)