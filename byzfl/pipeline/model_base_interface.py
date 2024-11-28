import collections

import torch

import byzfl.pipeline.models as models
from byzfl.utils.conversion import flatten_dict, unflatten_dict, unflatten_generator

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
        All this parameters should be passed in a dictionary that contains the following keys.
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

        if "fbn" in model_name:
            self.model = torch.nn.DataParallel(getattr(models, model_name)(params["nb_workers"])).to(self.device)
            self.use_fbn = True
        else:
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


    def use_federated_batch_norm(self):
        """
        Description
        -----------
        Getter to determine whether the model is using Federated Batch Normalization.

        Returns
        -------
        bool
            A boolean indicating whether the model is utilizing Federated Batch Normalization.
        """
        return self.use_fbn
