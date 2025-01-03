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
        if not isinstance(state_dict, dict):
            raise TypeError(f"'state_dict' must be of type dict, but got {type(state_dict).__name__}")
        self.model.load_state_dict(state_dict)