from torch.utils.data import DataLoader
import numpy as np
import torch, random

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
        if not isinstance(params["data_distribution_name"], str):
            raise TypeError("data_distribution_name must be a string")
        self.data_dist = params["data_distribution_name"]

        if "distribution_parameter" in params.keys():
            if not isinstance(params["distribution_parameter"], float):
                raise TypeError("distribution_parameter must be a float")
            if self.data_dist == "gamma_similarity_niid" and not (0.0 <= params["distribution_parameter"] <= 1.0):
                raise ValueError("distribution_parameter for gamma_similarity_niid must be between 0 and 1")
            self.distribution_parameter = params["distribution_parameter"]
        else:
            self.distribution_parameter = None

        if not isinstance(params["nb_honest"], int) or params["nb_honest"] <= 0:
            raise ValueError("nb_honest must be a positive integer")
        self.nb_honest = params["nb_honest"]

        if not (isinstance(params["data_loader"], torch.utils.data.DataLoader) or isinstance(params["data_loader"], torch.utils.data.Subset)):
            raise TypeError("data_loader must be an instance of torch.utils.data.DataLoader or torch.utils.data.Subset")
        self.data_loader = params["data_loader"]

        if not isinstance(params["batch_size"], int) or params["batch_size"] <= 0:
            raise ValueError("batch_size must be a positive integer")
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
        if isinstance(self.data_loader, torch.utils.data.DataLoader):
            idx = list(range(len(targets)))
        else:
            idx = self.data_loader.indices

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
        return np.array_split(idx, self.nb_honest)

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
        return np.array_split(sorted_idx, self.nb_honest)

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
        return [node_idx.astype(int) for node_idx in split_idx]

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
        return [list(idx[aux_idx[i]]) for i in range(len(aux_idx))]

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