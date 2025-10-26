from copy import deepcopy
import inspect

import numpy as np

from byzfl.poisoning import attacks
from byzfl.fed_framework.client import Client


class PoisoningClient(Client):
    """
    Initialization Parameters
    -------------------------
    client_params : dict
        A dictionary with the client configuration. See `Client` for more information.
    attack_params : dict
        A dictionary containing the configuration for the data poisoning attack. Must include:
        - `"p"`: float
            The proportion of samples to poison.
        - `"name"`: str
            The name of the attack to be executed (e.g., `"StaticLabelFlipping"`).
        - `"parameters"`: dict
            A dictionary of parameters for the specified attack, where keys are parameter names and values are their corresponding values.
        The attack must be callable with the following arguments:
        - model : torch.nn.Module
            The local model.
        - inputs : numpy.ndarray or torch.Tensor
            A collection of input features.
        - targets : numpy.ndarray or torch.Tensor
            A collection of targets.

    Methods
    -------
    apply_attack(inputs, targets)
        Applies the specified poisoning attack to the samples owned by the poisoning client and returns a tuple of corrupted inputs and targets.

    Calling the Instance
    --------------------
    Input Parameters
    ----------------
    inputs : numpy.ndarray or torch.Tensor
        A collection of input features.
    targets : numpy.ndarray or torch.Tensor
        A collection of targets.

    Returns
    -------
    tuple
        A tuple containing the inputs and targets poisoned by the attack with proportion `p`,
        each with the same data type as the inputs and targets provided to the instance.
    
    Notes
    -----
    - Contrary to `ByzantineClient`, a `PoisoningClient` is drop-in compatible with an honest
    `Client`.
    - The training losses are computed on the corrupted data.

    Examples
    --------
    Initialize the `PoisoningClient` with a specific attack:

    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from torchvision import datasets, transforms
    >>> from byzfl import PoisoningClient
    >>>
    >>> transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    >>> train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    >>> train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    >>>
    >>> params = {
    >>>     "model_name": "fc_mnist",
    >>>     "device": "cpu",
    >>>     "weight_decay": 1e-5,
    >>>     "optimizer_name": "Adam",
    >>>     "optimizer_params": {"betas": (0.9, 0.999)},
    >>>     "learning_rate": 0.01,
    >>>     "loss_name": "CrossEntropyLoss",
    >>>     "weight_decay": 0.0005,
    >>>     "milestones": [10, 20],
    >>>     "learning_rate_decay": 0.5,
    >>>     "momentum": 0.9,
    >>>     "training_dataloader": train_loader,
    >>>     "nb_labels": 10,
    >>> }
    >>>
    >>> attack = {
    >>>     "name": "StaticLabelFlipping",
    >>>     "parameters": {"permutation": {i: 9 - i for i in range(10)}},
    >>> }
    >>> poisoning_worker = PoisoningClient(params, attack)
    """

    def __init__(self, client_params, attack_params):
        """
        Initializes the `PoisoningClient` with the specified client parameters and
        attack configuration.

        Parameters
        ----------
        client_params : dict
            A dictionary with the client configuration. See `Client` for more information.
        attack_params : dict
            A dictionary with the attack configuration. Must include:
            - `"p"`: float
                The proportion of samples to poison.
            - `"name"`: str
                Name of the poisoning attack to execute.
            - `"parameters"`: dict
                Parameters for the specified attack.
        """

        super().__init__(client_params)
        self._init_poisoning_attack(attack_params)
    
    def _init_poisoning_attack(self, params):
        """
        Description
        -----------
        Initialize the `PoisoningClient` with the specified attack configuration.
        """
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        if "p" not in params or not isinstance(params["p"], float) or not 0.0 <= params["p"] <= 1:
            raise ValueError("p must be a float between 0.0 and 1.0")
        if "name" not in params or not isinstance(params["name"], str):
            raise TypeError("name must be a string")
        if "parameters" not in params or not isinstance(params["parameters"], dict):
            raise TypeError("parameters must be a dictionary")

        # Initialize the PoisoningClient instance
        self.no_attack = False
        self.p = params["p"]

        if params["name"] == "NoAttack":
            self.no_attack = True
            return

        self.attack = getattr(attacks, params["name"])
        signature_attack = inspect.signature(self.attack.__init__)

        filtered_parameters = {}
        for parameter in signature_attack.parameters.values():
            param_name = parameter.name
            if param_name in params["parameters"]:
                filtered_parameters[param_name] = params["parameters"][param_name]

            # If something goes wrong
            elif param_name == "p":
                filtered_parameters[param_name] = self.p

        self.attack = self.attack(**filtered_parameters)
    
    "@override"
    def _sample_train_batch(self):
        """
        Description
        -----------
        Retrieves the next batch of data from the training dataloader, after applying the
        poisoning attack. If the end of the dataset is reached, the dataloader is
        reinitialized to start from the beginning.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding target labels
            for the current batch, after applying the poisoning attack. The tensors are
            sent to `self.device`.
        """
        inputs, targets = super()._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        return self.apply_attack(inputs, targets)
    
    def apply_attack(self, inputs, targets):
        """
        Applies the specified poisoning attack to the samples owned by the poisoning client.

        Parameters
        ----------
        inputs : numpy.ndarray or torch.Tensor
            A collection of input features.
        targets : numpy.ndarray or torch.Tensor
            A collection of targets.

        Returns
        -------
        tuple
            A tuple containing the inputs and targets poisoned by the attack with proportion `p`,
            each with the same data type as the inputs and targets provided to the instance.
            If `no_attack` is `True`, the arguments are returned.
        """
        if len(inputs) != len(targets):
            raise ValueError("Inputs and targets lengths mismatch")

        inputs, targets = deepcopy((inputs, targets))
        if self.no_attack or self.attack == None or self.p == 0.0:
            return inputs, targets
        
        try:
            poisoned_inputs, poisoned_targets = self.attack(
                model=self.model,
                inputs=inputs,
                targets=targets,
            )
            return poisoned_inputs, poisoned_targets
        except Exception as e:
            raise NotImplementedError(
                f"Attack {self.attack} not implemented in PoisoningClient"
            ) from e