import inspect

from byzfl.attacks import attacks
from byzfl.aggregators import aggregators, preaggregators

class ByzantineClient:
    """
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
    Initialize the `ByzantineClient` with a specific attack and apply it to input vectors:

    >>> from byzfl import ByzantineClient
    >>> attack = {
    >>>     "name": "InnerProductManipulation",
    >>>     "f": 3,
    >>>     "parameters": {"tau": 3.0},
    >>> }
    >>> byz_worker = ByzantineClient(attack)

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
        Initializes the ByzantineClient with the specified attack configuration.

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

        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        if "f" not in params or not isinstance(params["f"], int) or params["f"] < 0:
            raise ValueError("f must be a non-negative integer")
        if "name" not in params or not isinstance(params["name"], str):
            raise TypeError("name must be a string")
        if "parameters" not in params or not isinstance(params["parameters"], dict):
            raise TypeError("parameters must be a dictionary")

        # Initialize the ByzantineClient instance
        self.f = params["f"]

        # Initialize the aggregator and pre-aggregators

        aggregator_info = params["parameters"]["aggregator_info"]
        pre_agg_list_info = params["parameters"]["pre_agg_list"]

        aggregator = getattr(aggregators, aggregator_info["name"])

        signature_agg = inspect.signature(aggregator.__init__)

        agg_parameters = {}

        for parameter in signature_agg.parameters.values():
            param_name = parameter.name
            if param_name in aggregator_info["parameters"]:
                agg_parameters[param_name] = aggregator_info["parameters"][param_name]
        
        aggregator = aggregator(**agg_parameters)

        pre_agg_list = []

        for pre_agg_info in pre_agg_list_info:

            pre_agg = getattr(
                preaggregators, 
                pre_agg_info["name"]
            )

            signature_pre_agg = inspect.signature(pre_agg.__init__)

            pre_agg_parameters = {}

            for parameter in signature_pre_agg.parameters.values():
                param_name = parameter.name
                if param_name in pre_agg_info["parameters"]:
                    pre_agg_parameters[param_name] = pre_agg_info["parameters"][param_name]

            pre_agg = pre_agg(**pre_agg_parameters)

            pre_agg_list.append(pre_agg)

        params["parameters"]["agg"] = aggregator
        params["parameters"]["pre_agg_list"] = pre_agg_list

        self.attack = getattr(attacks, params["name"])
        signature_attack = inspect.signature(self.attack.__init__)

        filtered_parameters = {}

        for parameter in signature_attack.parameters.values():
            param_name = parameter.name
            if param_name in params["parameters"]:
                filtered_parameters[param_name] = params["parameters"][param_name]

        self.attack = self.attack(**filtered_parameters)

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

        if not self.f < len(honest_vectors):
            raise ValueError(f"f must be smaller than the number of honest_vectors, but got f={self.f} and len(honest_vectors)={len(honest_vectors)}")
        if self.f == 0:
            return []

        # Generate the Byzantine vector by applying the attack
        byz_vector = self.attack(honest_vectors)

        # Return a list of the same Byzantine vector repeated `f` times
        return [byz_vector] * self.f