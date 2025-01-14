
import byzfl.fed_framework.models as models
from byzfl.utils.conversion import flatten_dict, unflatten_dict, unflatten_generator
import torch
import collections

class ModelBaseInterface(object):

    def __init__(self, params):
        # Input validation
        self._validate_params(params)

        # Initialize model
        model_name = params["model_name"]
        self.device = params["device"]
        self.model = torch.nn.DataParallel(getattr(models, model_name)()).to(self.device)

        # Initialize optimizer. If set to None, it means that the Client does not need this information 
        optimizer_name = params["optimizer_name"]
        if optimizer_name is not None:
            optimizer_params = params["optimizer_params"]
            optimizer_class = getattr(torch.optim, optimizer_name, None)
            if optimizer_class is None:
                raise ValueError(f"Optimizer '{optimizer_name}' is not supported by PyTorch.")
            
            self.optimizer = optimizer_class(
                self.model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                **optimizer_params
            )

            # Initialize scheduler
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=params["milestones"],
                gamma=params["learning_rate_decay"]
            )


    def _validate_params(self, params):
        """
        Validates the input parameters for correct types and values.

        Parameters
        ----------
        params : dict
            Dictionary of input parameters.
        """

        # Required keys for Server and Client
        required_keys = ["model_name", "device"]
        if params.get("isServer", False):
            # Required keys for Server, Optional for client
            required_keys += ["learning_rate", "weight_decay", "milestones", "learning_rate_decay"]
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required parameter: {key}")

        # Validate types and ranges
        if not isinstance(params["model_name"], str):
            raise TypeError("Parameter 'model_name' must be a string.")
        if not isinstance(params["device"], str):
            raise TypeError("Parameter 'device' must be a string.")
        if params["learning_rate"] is not None:
            if not isinstance(params["learning_rate"], float) or params["learning_rate"] <= 0:
                raise ValueError("Parameter 'learning_rate' must be a positive float.")
        if params["weight_decay"] is not None:
            if not isinstance(params["weight_decay"], float) or params["weight_decay"] < 0:
                raise ValueError("Parameter 'weight_decay' must be a non-negative float.")
        if params["milestones"] is not None:
            if not isinstance(params["milestones"], list) or not all(isinstance(x, int) for x in params["milestones"]):
                raise TypeError("Parameter 'milestones' must be a list of integers.")
        if params["learning_rate_decay"] is not None:
            if not isinstance(params["learning_rate_decay"], float) or params["learning_rate_decay"] <= 0 or params["learning_rate_decay"] > 1.0:
                raise ValueError("Parameter 'learning_rate_decay' must be a positive float smaller than 1.0.")

    def get_flat_parameters(self):
        """
        Returns model parameters in a flat array.

        Returns
        -------
        list
            Flat list of model parameters.
        """
        return flatten_dict(self.model.state_dict())

    def get_flat_gradients(self):
        """
        Returns model gradients in a flat array.

        Returns
        -------
        list
            Flat list of model gradients.
        """
        return flatten_dict(self.get_dict_gradients())

    def get_dict_parameters(self):
        """
        Returns model parameters in a dictionary format.

        Returns
        -------
        collections.OrderedDict
            Dictionary of model parameters.
        """
        return self.model.state_dict()

    def get_dict_gradients(self):
        """
        Returns model gradients in a dictionary format.

        Returns
        -------
        collections.OrderedDict
            Dictionary of model gradients.
        """
        new_dict = collections.OrderedDict()
        for key, value in self.model.named_parameters():
            new_dict[key] = value.grad
        return new_dict

    def set_parameters(self, flat_vector):
        """
        Sets model parameters using a flat array.

        Parameters
        ----------
        flat_vector : list
            Flat list of parameters to set.
        """
        new_dict = unflatten_dict(self.model.state_dict(), flat_vector)
        self.model.load_state_dict(new_dict)

    def set_gradients(self, flat_vector):
        """
        Sets model gradients using a flat array.

        Parameters
        ----------
        flat_vector : list
            Flat list of gradients to set.
        """
        new_dict = unflatten_generator(self.model.named_parameters(), flat_vector)
        for key, value in self.model.named_parameters():
            value.grad = new_dict[key].clone().detach()

    def set_model_state(self, state_dict):
        """
        Sets the state_dict of the model.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing model state.
        """
        self.model.load_state_dict(state_dict)