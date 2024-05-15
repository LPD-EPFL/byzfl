import collections

import torch

import models
from utils.conversion import flatten_dict, unflatten_dict, unflatten_generator

class ModelBaseInterface(object):
    """
    Description
    -----------
    This class is the basic interface with the methods that classes that are
    containers of a model needs to deal with the information exchange.

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

        self.model = getattr(models, model_name)().to(self.device)

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

        self.batch_norm_key_list = []
    
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
        batch_norm_stats = collections.OrderedDict()
        state_dict = self.model.state_dict()
        for key in self.batch_norm_key_list:
            batch_norm_stats[key] = state_dict[key]
        return batch_norm_stats

    def get_flat_batch_norm_stats(self):
        """
        Description
        ------------
        Get the batch norm stats of the model in a flatten array.

        Returns
        -------
        Array with the values of the batch norm stats.
        """
        batch_norm_stats = self.get_batch_norm_stats()
        return flatten_dict(batch_norm_stats)
    
    def get_batch_norm_keys(self):
        """
        Description
        ------------
        Getter with the list of the keys of the batch norm stats

        Returns
        -------
        List with the keys of the batch norm stats
        """
        return self.batch_norm_key_list
    
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
    
    def set_batch_norm_stats(self, flat_vector):
        """
        Description
        -----------
        Sets the model batch norm stats given a flat vector.

        Parameters
        ----------
        flat_vector : list
            Flat list with the parameters
        """
        batch_norm_stats = unflatten_dict(self.get_batch_norm_stats(), flat_vector)
        state_dict = self.model.state_dict()
        for key, item in batch_norm_stats.items():
            state_dict[key] = item
        self.set_model_state(state_dict)
    
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


    def update_model(self, gradients):
        """
        Description
        -----------
        Update the model aggregating the gradients given and do an step.

        Parameters
        ----------
        gradients : list
            Flat list with the gradients
        """
        agg_gradients = self.aggregate(gradients)
        self.set_gradients(agg_gradients)
        self.step()
    
    def update_batch_norm_stats(self, batch_norm_stats):
        """
        Description
        -----------
        Update the model aggregating the bath norm statistics given.

        Parameters
        ----------
        batch_norm_stats : list
            Flat list with the bath norm statistics
        """
        agg_stats = self.robust_aggregator.aggregate_batch_norm(batch_norm_stats)
        self.set_batch_norm_stats(agg_stats)

    def compute_batch_norm_keys(self):
        for key in self.model.state_dict().keys():
            if "running_mean" in key or "running_var" in key:
                self.batch_norm_key_list.append(key)
    
    def step(self):
        """
        Description
        -----------
        Do a step of the optimizer and the scheduler.
        """
        self.optimizer.step()
        self.scheduler.step()