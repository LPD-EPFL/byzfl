from byzfl.aggregators import aggregators
from byzfl.aggregators import preaggregators
import byzfl.attacks as attacks
import byzfl.pipeline.models as models
from byzfl.utils.conversion import flatten_dict, unflatten_dict, unflatten_generator
import inspect
import torch
import numpy as np
import collections

class ModelBaseInterface(object):
    """
    Description
    -----------
    This class serves as an abstract interface that defines the methods 
    required for classes that encapsulate a model. All subclasses that 
    contain a model should inherit from this class to ensure they implement 
    the necessary methods for handling model-related operations and information 
    exchange.

    Parameters
    ----------
        All these parameters should be passed in a dictionary that contains the following keys.
    model-name : str 
        Indicates the model to be used
    device : str
        Name of the device used
    learning-rate : float 
        Learning rate
    weight-decay : float 
        Regularization used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning-rate-decay : float 
        Rate decreases over time during training

    Methods
    --------
    """
    def __init__(self, params):
        model_name = params["model_name"]
        self.device = params["device"]
        self.model = torch.nn.DataParallel(getattr(models, model_name)()).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr = params["learning_rate"], 
            weight_decay = params["weight_decay"]
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones = params["milestones"],
            gamma = params["learning_rate_decay"]
        )

        self.batch_norm_keys = []
        self.running_mean_key_list = []
        self.running_var_key_list = []
    
    def get_flat_parameters(self):
        """
        Description
        -----------
        Get the gradients of the model in a flat array

        Returns
        -------
        List of the gradients
        """
        return flatten_dict(self.model.state_dict())
    
    def get_flat_gradients(self):
        """
        Description
        -----------
        Get the gradients of the model in a flat array

        Returns
        -------
        List of the gradients
        """
        return flatten_dict(self.get_dict_gradients())
    
    def get_dict_parameters(self):
        """
        Description
        ------------
        Get the gradients of the model in a dictionary.

        Returns
        -------
        Dicctionary where the keys are the name of the parameters
        and de values are the gradients.
        """
        return self.model.state_dict()

    def get_dict_gradients(self):
        """
        Description
        ------------
        Get the gradients of the model in a dictionary.

        Returns
        -------
        Dicctionary where the keys are the name of the parameters
        and the values are the gradients.
        """
        new_dict = collections.OrderedDict()
        for key, value in self.model.named_parameters():
            new_dict[key] = value.grad
        return new_dict
    
    def get_batch_norm_stats(self):
        """
        Description
        ------------
        Get the batch norm stats of the model in a dictionary.

        Returns
        -------
        Dicctionary where the keys are the name of the parameters
        of batch norm stats and their values.
        """
        running_mean_stats = collections.OrderedDict()
        state_dict = self.model.state_dict()
        for key in self.running_mean_key_list:
            running_mean_stats[key] = state_dict[key].clone()

        running_var_stats = collections.OrderedDict()
        for key in self.running_var_key_list:
            running_var_stats[key] = state_dict[key].clone()
        return running_mean_stats, running_var_stats

    def get_flat_batch_norm_stats(self):
        """
        Description
        ------------
        Get the batch norm stats of the model in a flatten array.

        Returns
        -------
        Array with the values of the batch norm stats.
        """
        running_mean_stats, running_var_stats = self.get_batch_norm_stats()
        return flatten_dict(running_mean_stats), flatten_dict(running_var_stats)
    
    def set_parameters(self, flat_vector):
        """
        Description
        -----------
        Sets the model parameters given a flat vector.

        Parameters
        ----------
        flat_vector : list 
            Flat list with the parameters
        """
        new_dict = unflatten_dict(self.model.state_dict(), flat_vector)
        self.model.load_state_dict(new_dict)

    def set_gradients(self, flat_vector):
        """
        Description
        -----------
        Sets the model gradients given a flat vector.

        Parameters
        ----------
        flat_vector : list
            Flat list with the parameters
        """
        new_dict = unflatten_generator(self.model.named_parameters(), flat_vector)
        for key, value in self.model.named_parameters():
            value.grad = new_dict[key].clone().detach()
    
    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Sets the state_dict of the model for the state_dict given by parameter.

        Parameters
        ----------
        state_dict : dict 
            State_dict from a model
        """
        self.model.load_state_dict(state_dict)

    def compute_batch_norm_keys(self):
        """
        Description
        -----------
        Compute batch normalization keys.

        """
        for key in self.model.state_dict().keys():
            if "running_mean" in key:
                self.running_mean_key_list.append(key)
                self.batch_norm_keys.append(key.split(".")[0])
            elif "running_var" in key:
                self.running_var_key_list.append(key)            
    
    def use_batch_norm(self):
        """
        Description
        -----------
        Getter to determine whether the model is using Batch Normalization.

        Returns
        -------
        bool
            A boolean indicating whether the model is utilizing Batch Normalization.
        """
        return len(self.batch_norm_keys) > 0

class Client(ModelBaseInterface):
    """
    Description
    -----------
    The `Client` class simulates a single honest node capable of training its local model, 
    sending gradients, and receiving the global model in every training round.

    Features
    --------
    - **Local Training**: Performs training on a local dataset using a specified model and optimizer.
    - **Gradient Computation**: Computes gradients for the local dataset.
    - **Momentum Support**: Supports momentum for gradient updates.
    - **Label Flipping Attack**: Optionally applies a label flipping attack during training.

    Initialization Parameters
    -------------------------
    params : dict
        A dictionary containing the configuration for the Client. Must include:

        - `"model_name"`: str
            Name of the model to be used.
        - `"device"`: str
            Device for computation (e.g., 'cpu' or 'cuda').
        - `"learning_rate"`: float
            Learning rate for the optimizer.
        - `"loss_name"`: str
            Loss function name (e.g., 'CrossEntropyLoss').
        - `"weight_decay"`: float
            Weight decay for regularization.
        - `"milestones"`: list
            Milestones for learning rate decay.
        - `"learning_rate_decay"`: float
            Learning rate decay factor.
        - `"LabelFlipping"`: bool
            A boolean flag that, when set to True, enables the LabelFlipping attack. This attack flips the class labels of the training data to their opposing classes.
        - `"momentum"`: float
            Momentum for the optimizer.
        - `"training_dataloader"`: DataLoader
            DataLoader for the training data.
        - `"nb_labels"`: int
            Number of labels in the dataset.
        - `"nb_steps"`: int
            Number of training steps.

    Methods
    -------
    compute_gradients
        Computes gradients for the local dataset.
    get_flat_flipped_gradients
        Returns the gradients of the model with flipped targets in a flat array.
    get_flat_gradients_with_momentum
        Returns flattened gradients with momentum applied.
    get_loss_list
        Returns the list of training losses.
    get_train_accuracy
        Returns the training accuracy per batch.
    set_model_state(state_dict)
        Updates the model's state dictionary.

    Examples
    --------
    Initialize the `Client` class with an MNIST data loader:

    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from torchvision import datasets, transforms
    >>> from byzfl.pipeline import Client

    >>> # Fix the random seed of torch
    >>> SEED = 42
    >>> torch.manual_seed(SEED)
    >>> torch.cuda.manual_seed(SEED)
    >>> torch.backends.cudnn.deterministic = True
    >>> torch.backends.cudnn.benchmark = False
    >>> # Define the training data loader using MNIST
    >>> transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    >>> train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    >>> train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    >>> # Define client parameters
    >>> client_params = {
    >>>     "model_name": "cnn_mnist", 
    >>>     "device": "cpu", 
    >>>     "learning_rate": 0.01, 
    >>>     "loss_name": "CrossEntropyLoss", 
    >>>     "weight_decay": 0.0005, 
    >>>     "milestones": [10, 20], 
    >>>     "learning_rate_decay": 0.5, 
    >>>     "LabelFlipping": True, 
    >>>     "momentum": 0.9, 
    >>>     "training_dataloader": train_loader, 
    >>>     "nb_labels": 10, 
    >>>     "nb_steps": len(train_loader)
    >>> }

    >>> # Initialize the Client
    >>> client = Client(client_params)

    Compute gradients for the current training batch:

    >>> # Compute first gradient
    >>> client.compute_gradients()
    >>> # Initial train accuracy
    >>> client.get_train_accuracy()[0]
    >>> # Get first flipped gradient
    >>> client.get_flat_flipped_gradients()
    tensor([-0.0005,  0.0008,  0.0027,  ...,  0.0732,  0.0722, -0.0281])
    0.171875
    tensor([-0.0002, -0.0027, -0.0032,  ..., -0.0675, -0.0215, -0.0125])

    """

    def __init__(self, params):
        super().__init__({
            "model_name": params["model_name"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "nb_workers": params.get("nb_workers", 1)
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
        self.loss_list = np.array([0.0] * params["nb_steps"])
        self.train_acc_list = np.array([0.0] * params["nb_steps"])

        self.SGD_step = 0

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
        self.loss_list[self.SGD_step] = loss.item()
        loss.backward()

        # Compute train accuracy
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        acc = correct / total
        self.train_acc_list[self.SGD_step] = acc

        self.SGD_step += 1

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
        self.model.load_state_dict(state_dict)


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


class ByzantineClient:
    """
    Description
    -----------
    The `ByzantineClient` class is responsible for simulating Byzantine behavior in distributed machine learning 
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