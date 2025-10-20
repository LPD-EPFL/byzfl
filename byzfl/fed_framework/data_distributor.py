import numpy as np
import torch, random
from torch.utils.data import DataLoader

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
                raise ValueError("distribution_parameter for gamma_similarity_niid must be between 0.0 and 1.0")
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
        elif self.data_dist == "dirichlet_niid_modified":
            split_idx= self.dirichlet_niid_modified_idx(targets, idx) #modified to account for minimal per node-batch size.
        elif self.data_dist == "extreme_niid_modified":
            split_idx = self.extreme_niid_modified_idx(targets, idx)
        
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
    
    def extreme_niid_modified_idx(self, targets, idx):
        """
        Creates an extremely non-IID partition of the dataset, and corrects the former implementation by avoiding that 
        classes overlap over several clients.

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
        # sorted_idx = np.array(sorted(zip(targets[idx], idx)))[:, 1]
        # return np.array_split(sorted_idx, self.nb_honest)
        
      
        classes=torch.unique(targets)
        class_idx_dict = {target: idx_target for idx_target, target in enumerate(classes)} #allows to convert labels to consecutive integers
        
        c=len(class_idx_dict) #nb of classes
        
        aux_idx = [idx[np.where(targets[idx] == k)[0]] for k in classes] #stores indices corresponding to each class

        if c>= self.nb_honest: #more classes than clients: some clients have multiple classes.
            partition = [[] for _ in range(self.nb_honest)]
            for idx_target in class_idx_dict.values():
                node_idx = idx_target % self.nb_honest
                partition[node_idx].append(aux_idx[idx_target])
            return partition
        
        elif c<self.nb_honest:
            raise ValueError("there must be at least as much classes as honest nodes for a dirichlet distribution.")
            
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
        sample = np.random.dirichlet(np.repeat(self.distribution_parameter, self.nb_honest), size=c) #here, we have a (c, nb_honest) matrix, whith each each row summing to 1. This clearly does not garantee that each node has data.(it should e the rows)
        p = np.cumsum(sample, axis=1)[:, :-1]
        aux_idx = [np.where(targets[idx] == k)[0] for k in range(c)]
        aux_idx = [np.split(aux_idx[k], (p[k] * len(aux_idx[k])).astype(int)) for k in range(c)]
        aux_idx = [np.concatenate([aux_idx[i][j] for i in range(c)]) for j in range(self.nb_honest)]
        idx = np.array(idx)
        return [list(idx[aux_idx[i]]) for i in range(len(aux_idx))]
    
    def dirichlet_niid_modified_idx(self, targets, idx, min_size=10):
        """
        Creates a modified Dirichlet non-IID partition of the dataset, as in *mean is more robust*.
        This partition ensures that each node has data, by adjusting the generated proportions until each node has at least min_size samples.

        Parameters
        ----------
        targets : numpy.ndarray
            Array of dataset targets (labels).
        idx : numpy.ndarray
            Array of dataset indices corresponding to the targets.
            
        min_size: int
            default is 10, as in the literature.

        Returns
        -------
        list[numpy.ndarray]
            A list of arrays where each array contains indices for one client.
        """

        current_min_size=0
        data_size = len(targets)
        c = len(torch.unique(targets))
        idx_classes = [np.where(targets[idx] == k)[0] for k in range(c)]

        
        partition = [[] for _ in range(self.nb_honest)]
        while current_min_size < min_size:
            partition = [[] for _ in range(self.nb_honest)]
            for k in range(c):
                idx_k = idx_classes[k]
                random.shuffle(idx_k)
                
                proportions = np.random.dirichlet(np.repeat(self.distribution_parameter, self.nb_honest))
                # using the proportions from dirichlet, only select those nodes having data amount less than average
                proportions = np.array(
                    [p * (len(idx_j) < data_size / self.nb_honest) for p, idx_j in zip(proportions, partition)])
                # scale proportions
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                partition = [idx_j + idx.tolist() for idx_j, idx in zip(partition, np.split(idx_k, proportions))]
                current_min_size = min([len(idx_j) for idx_j in partition])
        return partition

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