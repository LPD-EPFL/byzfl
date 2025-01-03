import torch
from byzfl.fed_framework import ModelBaseInterface
from byzfl.fed_framework import RobustAggregator

class Server(ModelBaseInterface):
    
    def __init__(self, params):
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["test_loader"], torch.utils.data.DataLoader):
            raise TypeError(f"'test_loader' must be a DataLoader, but got {type(params['test_loader']).__name__}")

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
            if not isinstance(params["validation_loader"], torch.utils.data.DataLoader):
                raise TypeError(f"'validation_loader' must be a DataLoader, but got {type(params['validation_loader']).__name__}")

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