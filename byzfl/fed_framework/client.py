import torch
from byzfl.fed_framework import ModelBaseInterface
from byzfl.utils.conversion import flatten_dict

class Client(ModelBaseInterface):

    def __init__(self, params):
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["loss_name"], str):
            raise TypeError(f"'loss_name' must be of type str, but got {type(params['loss_name']).__name__}")
        if not isinstance(params["LabelFlipping"], bool):
            raise TypeError(f"'LabelFlipping' must be of type bool, but got {type(params['LabelFlipping']).__name__}")
        if not isinstance(params["nb_labels"], int) or not params["nb_labels"] > 1:
            raise ValueError(f"'nb_labels' must be an integer greater than 1")
        if not isinstance(params["momentum"], float) or not 0 <= params["momentum"] < 1:
            raise ValueError(f"'momentum' must be a float in the range [0, 1)")
        if not isinstance(params["training_dataloader"], torch.utils.data.DataLoader):
            raise TypeError(f"'training_dataloader' must be a DataLoader, but got {type(params['training_dataloader']).__name__}")

        # Initialize Client instance
        super().__init__({
            # Required parameters
            "model_name": params["model_name"],
            "device": params["device"],
            # Optional parameters
            "learning_rate": params.get("learning_rate", None),
            "weight_decay": params.get("weight_decay", None),
            "milestones": params.get("milestones", None),
            "learning_rate_decay": params.get("learning_rate_decay", None),
            "optimizer_name": params.get("optimizer_name", None),
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
        Description
        -----------
        Retrieves the next batch of data from the training dataloader. If the 
        end of the dataset is reached, the dataloader is reinitialized to start 
        from the beginning.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding target labels for the current batch.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Description
        -----------
        Computes the gradients of the local model's loss function for the 
        current training batch. If the `LabelFlipping` attack is enabled, 
        gradients for flipped targets are computed and stored separately. 
        Additionally, the training loss and accuracy for the batch are 
        computed and recorded.
        """
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)
            self._backward_pass(inputs, targets_flipped)
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        train_loss_value = self._backward_pass(inputs, targets, train_acc=True)
        self.loss_list.append(train_loss_value)

    def _backward_pass(self, inputs, targets, train_acc=False):
        """
        Description
        -----------
        Performs a backward pass through the model to compute gradients for 
        the given inputs and targets. Optionally computes training accuracy 
        for the batch.

        Parameters
        ----------
        inputs : torch.Tensor
            The input data for the batch.
        targets : torch.Tensor
            The target labels for the batch.
        train_acc : bool, optional
            If True, computes and stores the training accuracy for the batch. 
            Default is False.

        Returns
        -------
        float
            The loss value for the current batch.
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        loss.backward()

        if train_acc:
            # Compute and store train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            self.train_acc_list.append(acc)

        return loss_value

    def get_flat_flipped_gradients(self):
        """
        Description
        -----------
        Retrieves the gradients computed using flipped targets as a flat array.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            A flat array containing the gradients for the model parameters 
            when trained with flipped targets.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Description
        -----------
        Computes the gradients with momentum applied and returns them as a 
        flat array.

        Returns
        -------
        torch.Tensor
            A flat array containing the gradients with momentum applied.
        """
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(
            self.get_flat_gradients(),
            alpha=1 - self.momentum
        )
        return self.momentum_gradient

    def get_loss_list(self):
        """
        Description
        -----------
        Retrieves the list of training losses recorded over the course of 
        training.

        Returns
        -------
        list
            A list of float values representing the training losses for each 
            batch.
        """
        return self.loss_list

    def get_train_accuracy(self):
        """
        Description
        -----------
        Retrieves the training accuracy for each batch processed during 
        training.

        Returns
        -------
        list
            A list of float values representing the training accuracy for each 
            batch.
        """
        return self.train_acc_list

    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Updates the state of the model with the provided state dictionary. 
        This method is used to load a saved model state or update 
        the global model in a federated learning context.
        Typically, this method can be used to synchronize clients with the global model.

        Parameters
        ----------
        state_dict : dict
            The state dictionary containing model parameters and buffers.

        Raises
        ------
        TypeError
            If `state_dict` is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"'state_dict' must be of type dict, but got {type(state_dict).__name__}")
        self.model.load_state_dict(state_dict)