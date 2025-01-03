from byzfl.aggregators import aggregators
from byzfl.aggregators import preaggregators
import byzfl.attacks as attacks
import byzfl.fed_framework.models as models
from byzfl.utils.conversion import flatten_dict, unflatten_dict, unflatten_generator
from byzfl.utils.misc import *
import inspect
import torch
import numpy as np
import collections
import random
from torch.utils.data import DataLoader

class ModelBaseInterface(object):

    def __init__(self, params):
        # Input validation
        self._validate_params(params)

        # Initialize model
        model_name = params["model_name"]
        self.device = params["device"]
        self.model = torch.nn.DataParallel(getattr(models, model_name)()).to(self.device)

        # Initialize optimizer, set default to SGD if optimizer_name is not provided
        optimizer_name = params.get("optimizer_name", "SGD")
        optimizer_params = params.get("optimizer_params", {})
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
        required_keys = ["model_name", "device", "learning_rate", "weight_decay", "milestones", "learning_rate_decay"]
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required parameter: {key}")

        # Validate types and ranges
        if not isinstance(params["model_name"], str):
            raise TypeError("Parameter 'model_name' must be a string.")
        if not isinstance(params["device"], str):
            raise TypeError("Parameter 'device' must be a string.")
        if not isinstance(params["learning_rate"], float) or params["learning_rate"] <= 0:
            raise ValueError("Parameter 'learning_rate' must be a positive float.")
        if not isinstance(params["weight_decay"], float) or params["weight_decay"] < 0:
            raise ValueError("Parameter 'weight_decay' must be a non-negative float.")
        if not isinstance(params["milestones"], list) or not all(isinstance(x, int) for x in params["milestones"]):
            raise TypeError("Parameter 'milestones' must be a list of integers.")
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

class Client(ModelBaseInterface):

    def __init__(self, params):
        # Check for correct types and values in params
        check_type(params, "params", dict)
        check_type(params["loss_name"], "loss_name", str)
        check_type(params["LabelFlipping"], "LabelFlipping", bool)
        check_type(params["nb_labels"], "nb_labels", int)
        check_greater_than_value(params["nb_labels"], "nb_labels", 1)
        check_type(params["momentum"], "momentum", float)
        check_greater_than_or_equal_value(params["momentum"], "momentum", 0)
        check_smaller_than_value(params["momentum"], "momentum", 1)
        check_type(params["training_dataloader"], "training_dataloader", torch.utils.data.DataLoader)

        # Initialize Client instance
        super().__init__({
            "model_name": params["model_name"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "optimizer_name": params["optimizer_name"],
            "optimizer_params": params.get("optimizer_params", {}),
        })

        self.criterion = getattr(torch.nn, params["loss_name"])()
        self.gradient_LF = 0
        self.labelflipping = params["LabelFlipping"]
        self.nb_labels = params["nb_labels"]
        self.momentum = params["momentum"]
        self.momentum_gradient = torch.zeros_like(
            torch.cat(tuple(
                tensor.view(-1) 
                for tensor in self.model.parameters()
            )),
            device=params["device"]
        )
        self.training_dataloader = params["training_dataloader"]
        self.train_iterator = iter(self.training_dataloader)
        self.loss_list = list()
        self.train_acc_list = list()

    def _sample_train_batch(self):
        """
        Private function to get the next data from the dataloader.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Computes the gradients of the local model loss function.
        """
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            self.model.zero_grad()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets_flipped)
            loss.backward()
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        self.loss_list.append(loss.item())
        loss.backward()

        # Compute train accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        acc = correct / total
        self.train_acc_list.append(acc)

    def get_flat_flipped_gradients(self):
        """
        Returns the gradients of the model with flipped targets in a flat array.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Returns the gradients with momentum applied in a flat array.
        """
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(
            self.get_flat_gradients(),
            alpha=1 - self.momentum
        )

        return self.momentum_gradient

    def get_loss_list(self):
        """
        Returns the list of computed losses over training.
        """
        return self.loss_list

    def get_train_accuracy(self):
        """
        Returns the training accuracy per batch.
        """
        return self.train_acc_list

    def set_model_state(self, state_dict):
        """
        Updates the model state with the provided state dictionary.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of a model.
        """
        check_type(state_dict, "state_dict", dict)
        self.model.load_state_dict(state_dict)


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
        check_type(aggregator_info, "aggregator_info", dict)
        check_type(pre_agg_list, "pre_agg_list", list)
        for pre_agg in pre_agg_list:
            check_type(pre_agg, "pre_agg", dict)

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
        check_type(params, "params", dict)
        check_type(params["f"], "f", int)
        check_greater_than_or_equal_value(params["f"], "f", 0)
        check_type(params["name"], "name", str)
        check_type(params["parameters"], "parameters", dict)

        # Initialize the ByzantineClient instance
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

        check_smaller_than_value(self.f, "f", len(honest_vectors))
        if self.f == 0:
            return []

        # Generate the Byzantine vector by applying the attack
        byz_vector = self.attack(honest_vectors)

        # Return a list of the same Byzantine vector repeated `f` times
        return [byz_vector] * self.f


class Server(ModelBaseInterface):
    
    def __init__(self, params):
        # Check for correct types and values in params
        check_type(params, "params", dict)
        check_type(params["test_loader"], "test_loader", torch.utils.data.DataLoader)

        # Initialize the Server instance
        super().__init__({
            "device": params["device"],
            "model_name": params["model_name"],
            "optimizer_name": params["optimizer_name"],
            "optimizer_params": params.get("optimizer_params", {}),
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
        })
        self.robust_aggregator = RobustAggregator(params["aggregator_info"], params["pre_agg_list"])
        self.test_loader = params["test_loader"]
        self.validation_loader = params.get("validation_loader")
        if self.validation_loader is not None:
            check_type(self.validation_loader, "validation_loader", torch.utils.data.DataLoader)
        self.model.eval()

    def aggregate(self, vectors):
        """
        Description
        -----------
        Aggregates input vectors using the configured robust aggregator.

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A collection of input vectors.

        Returns
        -------
        Aggregated output vector.
        """
        return self.robust_aggregator.aggregate_vectors(vectors)

    def update_model(self, gradients):
        """
        Description
        -----------
        Updates the global model by aggregating gradients and performing an optimization step.

        Parameters
        ----------
        gradients : list
            List of gradients to aggregate and apply.
        """
        aggregate_gradient = self.aggregate(gradients)
        self.set_gradients(aggregate_gradient)
        self.step()

    def step(self):
        """
        Description
        -----------
        Performs a single optimization step for the global model.
        """
        self.optimizer.step()
        self.scheduler.step()

    def get_model(self):
        """
        Description
        -----------
        Retrieves the current global model.

        Returns
        -------
        torch.nn.Module
            The current global model.
        """
        return self.model

    def _compute_accuracy(self, data_loader):
        total = 0
        correct = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct / total

    def compute_validation_accuracy(self):
        """
        Description
        -----------
        Computes the accuracy of the global model on the validation dataset.

        Returns
        -------
        float
            Validation accuracy.
        """
        if self.validation_loader is None:
            print("Validation Data Loader is not set.")
            return
        return self._compute_accuracy(self.validation_loader)

    def compute_test_accuracy(self):
        """
        Description
        -----------
        Computes the accuracy of the global model on the test dataset.

        Returns
        -------
        float
            Test accuracy.
        """
        return self._compute_accuracy(self.test_loader)


class DataDistributor:
    """
    Initialization Parameters
    -------------------------
    params : dict
        A dictionary containing the configuration for the data distributor. Must include:

        - `"data_distribution_name"` : str  
            Name of the data distribution strategy (`"iid"`, `"gamma_similarity_niid"`, etc.).
        - `"distribution_parameter"` : float  
            Parameter for the data distribution strategy (e.g., gamma or alpha).
        - `"nb_honest"` : int  
            Number of honest clients to split the dataset among.
        - `"data_loader"` : DataLoader  
            The data loader of the dataset to be distributed.
        - `"batch_size"` : int  
            Batch size for the generated dataloaders.

    Methods
    -------
    - **`split_data()`**:  
      Splits the dataset into dataloaders based on the specified distribution strategy.

    Example
    -------
    >>> from torchvision import datasets, transforms
    >>> from torch.utils.data import DataLoader
    >>> from byzfl import DataDistributor
    >>> transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    >>> dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    >>> data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    >>> params = {
    >>>     "data_distribution_name": "dirichlet_niid",
    >>>     "distribution_parameter": 0.5,
    >>>     "nb_honest": 5,
    >>>     "data_loader": data_loader,
    >>>     "batch_size": 64,
    >>> }
    >>> distributor = DataDistributor(params)
    >>> dataloaders = distributor.split_data()
    """

    def __init__(self, params):
        """
        Initializes the DataDistributor.

        Parameters
        ----------
        params : dict
            A dictionary containing configuration for the data distribution. Must include:
            - "data_distribution_name" (str): The type of data distribution (e.g., "iid", "gamma_similarity_niid").
            - "distribution_parameter" (float): Parameter specific to the chosen distribution.
            - "nb_honest" (int): Number of honest participants.
            - "data_loader" (DataLoader): The DataLoader of the dataset to be split.
            - "batch_size" (int): Batch size for the resulting DataLoader objects.
        """

        # Type and Value checking, and initialization of the DataDistributor class
        check_type(params["data_distribution_name"], "data_distribution_name", str)
        self.data_dist = params["data_distribution_name"]

        if "distribution_parameter" in params.keys():
            check_type(params["distribution_parameter"], "distribution_parameter", float)
            if self.data_dist == "gamma_similarity_niid":
                check_smaller_than_or_equal_value(params["distribution_parameter"], "distribution_parameter", 1.0)
            self.distribution_parameter = params["distribution_parameter"]
        else:
            self.distribution_parameter = None

        check_type(params["nb_honest"], "nb_honest", int)
        check_greater_than_value(params["nb_honest"], "nb_honest", 0)
        self.nb_honest = params["nb_honest"]

        check_type(params["data_loader"], "data_loader", torch.utils.data.DataLoader)
        self.data_loader = params["data_loader"]

        check_type(params["batch_size"], "batch_size", int)
        check_greater_than_value(params["batch_size"], "batch_size", 0)
        self.batch_size = params["batch_size"]        

    def split_data(self):
        """
        Splits the dataset according to the specified distribution strategy.

        Returns
        -------
        list[DataLoader]
            A list of DataLoader objects for each client.

        Raises
        ------
        ValueError
            If the specified data distribution name is invalid.
        """
        targets = self.data_loader.dataset.targets
        #idx = self.dataset.dataset.indices
        idx = list(range(len(targets)))

        if self.data_dist == "iid":
            split_idx = self.iid_idx(idx)
        elif self.data_dist == "gamma_similarity_niid":
            split_idx = self.gamma_niid_idx(targets, idx)
        elif self.data_dist == "dirichlet_niid":
            split_idx = self.dirichlet_niid_idx(targets, idx)
        elif self.data_dist == "extreme_niid":
            split_idx = self.extreme_niid_idx(targets, idx)
        else:
            raise ValueError(f"Invalid value for data_dist: {self.data_dist}")

        return self.idx_to_dataloaders(split_idx)

    def iid_idx(self, idx):
        """
        Splits indices into IID (independent and identically distributed) partitions.

        Parameters
        ----------
        idx : numpy.ndarray
            Array of dataset indices.

        Returns
        -------
        list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.
        """
        random.shuffle(idx)
        split_idx = np.array_split(idx, self.nb_honest)
        return split_idx

    def extreme_niid_idx(self, targets, idx):
        """
        Creates an extremely non-IID partition of the dataset.

        Parameters
        ----------
        targets : numpy.ndarray
            Array of dataset targets (labels).
        idx : numpy.ndarray
            Array of dataset indices corresponding to the targets.

        Returns
        -------
        list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.
        """
        if len(idx) == 0:
            return list([[]] * self.nb_honest)
        sorted_idx = np.array(sorted(zip(targets[idx], idx)))[:, 1]
        split_idx = np.array_split(sorted_idx, self.nb_honest)
        return split_idx

    def gamma_niid_idx(self, targets, idx):
        """
        Creates a gamma-similarity non-IID partition of the dataset.

        Parameters
        ----------
        targets : numpy.ndarray
            Array of dataset targets (labels).
        idx : numpy.ndarray
            Array of dataset indices corresponding to the targets.

        Returns
        -------
        list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.
        """
        nb_similarity = int(len(idx) * self.distribution_parameter)
        iid = self.iid_idx(idx[:nb_similarity])
        niid = self.extreme_niid_idx(targets, idx[nb_similarity:])
        split_idx = [np.concatenate((iid[i], niid[i])) for i in range(self.nb_honest)]
        split_idx = [node_idx.astype(int) for node_idx in split_idx]
        return split_idx

    def dirichlet_niid_idx(self, targets, idx):
        """
        Creates a Dirichlet non-IID partition of the dataset.

        Parameters
        ----------
        targets : numpy.ndarray
            Array of dataset targets (labels).
        idx : numpy.ndarray
            Array of dataset indices corresponding to the targets.

        Returns
        -------
        list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.
        """
        c = len(torch.unique(targets))
        sample = np.random.dirichlet(np.repeat(self.distribution_parameter, self.nb_honest), size=c)
        p = np.cumsum(sample, axis=1)[:, :-1]
        aux_idx = [np.where(targets[idx] == k)[0] for k in range(c)]
        aux_idx = [np.split(aux_idx[k], (p[k] * len(aux_idx[k])).astype(int)) for k in range(c)]
        aux_idx = [np.concatenate([aux_idx[i][j] for i in range(c)]) for j in range(self.nb_honest)]
        idx = np.array(idx)
        aux_idx = [list(idx[aux_idx[i]]) for i in range(len(aux_idx))]
        return aux_idx

    def idx_to_dataloaders(self, split_idx):
        """
        Converts index splits into DataLoader objects.

        Parameters
        ----------
        split_idx : list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.

        Returns
        -------
        list[DataLoader]
            A list of DataLoader objects for each client.
        """
        data_loaders = []
        for i in range(len(split_idx)):
            subset = torch.utils.data.Subset(self.data_loader.dataset, split_idx[i])
            data_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            data_loaders.append(data_loader)
        return data_loaders