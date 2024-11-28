import torch
import numpy as np

from byzfl.pipeline.model_base_interface import ModelBaseInterface
from byzfl.utils.conversion import flatten_dict

class Client(ModelBaseInterface):
    """
    Description
    -----------
    This class simulates one honest node which is able to train 
    his local model to send the gradients and get the global model every round.

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    model_name : str 
        Name of the model to be used
    device : str 
        Name of the device to be used
    learning_rate : float 
        Learning rate
    loss_name : str
        Name of the loss to be used
    weight_decay : float 
        Regularization used
    milestones : list 
        List of the milestones, where the learning rate
        decay should be applied
    learning_rate_decay : float
        Rate decreases over time during training
    attack_name : str 
        Name of the attack to be used
    momentum : float 
        Momentum
    training_dataloader : Dataloader
        Traning dataloader
    nb_labels : int 
        Number of labels in the dataset
    nb_steps : int
        Number of steps in the training

    Methods
    --------
    """
    def __init__(self, params):
        super().__init__({
            "model_name": params["model_name"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "nb_workers": params["nb_workers"]
        })

        self.criterion = getattr(torch.nn, params["loss_name"])()

        self.gradient_LF = 0
        self.labelflipping = params["attack_name"] == "LabelFlipping"
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
        self.loss_list = np.array([0.0]*params["nb_steps"])
        self.train_acc_list = np.array([0.0]*params["nb_steps"])

        self.SDG_step = 0

    def _sample_train_batch(self):
        """
        Description
        -----------
        Private function to get the next data from the dataloader
        """
        try:
            return next(self.training_dataloader)
        except:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Description
        -----------
        Function where the client compute their gradients 
        of their model loss function.
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
        self.loss_list[self.SDG_step] = loss.item()
        loss.backward()

        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        acc = correct/total
        self.train_acc_list[self.SDG_step] = acc

        self.SDG_step += 1
    
    def get_flat_flipped_gradients(self):
        """
        Description
        -----------
        Get the gradients of the model with their targets
        flipped in a flat array.

        Returns
        -------
        List of the gradients.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Description
        ------------
        Get the gradients of the model applying the momentum 
        in a flat array.

        Returns
        -------
        List with the gradiends with momentum applied.
        """
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(
            self.get_flat_gradients(),
            alpha=1-self.momentum
        )

        return self.momentum_gradient
    
    def get_loss_list(self):
        """
        Description
        ------------
        Get the loss list of the client

        Returns
        -------
        List with the losses that have been computed over the training.
        """
        return self.loss_list
    
    def get_train_accuracy(self):
        """
        Description
        ------------
        Get the batch train accuracy list of the client

        Returns
        -------
        List with the accuracies that have been computed over the training.
        """
        return self.train_acc_list
    
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
