import torch

from robust_aggregators import RobustAggregator
from model_base_interface import ModelBaseInterface

class Server(ModelBaseInterface):
    """
    Description
    -----------
    This class simulates the central server of our environment
    where the global model is updated.

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    model-name : str
        Indicates the model to be used
    dataloader : Dataloader
        Dataloader with the validation set to compute the accuracy of the global model
    test_dataloader : Dataloader
        Dataloader with the validation set to compute the test accuracy of the global model
    device : str 
        Name of the device used
    bit_precision : int
        How many bits will be displayed in the accuracy
    learning-rate : float 
        Learning rate
    weight-decay : float 
        Weight decay used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning-rate-decay : float
        Lerning rate decay used
    dataset-name : str 
        Name of the dataset used
    aggregator-info : dict 
        Dictionary with the keys "name" and "parameters" defined.
    pre-agg-info : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.

    Methods
    -------
    """
    def __init__(self, params):
        super().__init__({
            "model_name": params["model_name"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"]
        })
        self.robust_aggregator = RobustAggregator(
            params["aggregator_info"],
            params["pre_agg_list"]
        )
        self.validation_loader = params["dataloader"]
        self.test_loader = params["test_dataloader"]
        self.bit_precision = params["bit_precision"]
        self.model.eval()

    def aggregate(self, vectors):
        """
        Description
        -----------
        Aggregate vector using robust aggregation

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) where each
            row represents a vector.
        
        Returns
        -------
        list or ndarray or torch.Tensor
            The average vector of the input. The data type of the output will be the same as the input.
        """
        return self.robust_aggregator.aggregate(vectors)
    
    @torch.no_grad()
    def _compute_accuracy(self, dataloader):
        """
        Description
        -----------
        Compute the accuracy using the test set of the model

        Returns
        -------
        A float with the accuracy value
        """
        total = 0
        correct = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return round(correct/total, self.bit_precision)
    
    def compute_validation_accuracy(self):
        return self._compute_accuracy(self.validation_loader)
    
    def compute_test_accuracy(self):
        return self._compute_accuracy(self.test_loader)