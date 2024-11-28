from byzfl.pipeline.client import Client
from byzfl.pipeline.byzantine_worker import ByzantineWorker

class ComputeCluster(object):
    """
    Description
    -----------
    This class is the box that contains all the nodes in our
    system (honest and byzantine) and it's responsability is
    manage all this nodes and the information transference.

    Parameters
    -----------
        All this parameters should be passed in a dictionary that contains the following keys.
    nb_workers : int 
        Number of workers
    nb_byz : int
        Number of byzantine nodes
    model-name : str 
        Indicates the model to be used
    device : str 
        Name of the device used
    learning_rate : float 
        Learning rate
    loss_name : str 
        Loss name
    weight_decay : float 
        Regularization used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning_rate_decay : float 
        Rate decreases over time during training
    momentum : float 
        Momentum
    attack_name : str 
        Name of the attack to be used
    attack_parameters : dict 
        Dictionary with the parameters of the attack where every key is
        the name of the paramater and their value is the value of 
        the parameter
    attack_optimizer_name : (str, optional)
        Name of the optimizer to be used to find the best attack parameters
    attack_optimizer_parameters : (dict, optional)
        Dictionary with the parameters of the optimizer where every key 
        is the name of the paramater and their value is the value of 
        the parameter.
    aggregator_info : dict
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_list : list 
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    dataloaders : list of Dataloader
        List of Dataloader with the train set to every client.
    nb-labels : int
        Number of labels in the dataset

    Methods
    -------
    """
    def __init__(self, params):
        params_client = {
            "model_name": params["model_name"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "loss_name": params["loss_name"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "attack_name": params["attack_name"],
            "momentum": params["momentum"],
            "nb_labels": params["nb_labels"],
            "nb_workers": params["nb_workers"],
            "nb_steps": params["nb_steps"],
        }

        training_dataloader = params["dataloaders"]

        nb_honest = params["nb_workers"] - params["nb_byz"]

        self.client_list = [
            Client({
                **params_client, 
                "training_dataloader": training_dataloader[idx]
            }) for idx in range(nb_honest)
        ]
        self.labelflipping = False

        if params["attack_name"] == "LabelFlipping":
            self.labelflipping = True
            params["attack_name"] = "NoAttack"
        
        params_byz_worker = {
            "nb_byz": params["nb_byz"],
            "attack_name": params["attack_name"],
            "attack_parameters": params["attack_parameters"],
            "attack_optimizer_name": params["attack_optimizer_name"],
            "attack_optimizer_parameters": params["attack_optimizer_parameters"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"]
        }

        self.byz_worker = ByzantineWorker(params_byz_worker)
    
    def get_gradients(self):
        """
        Description
        -----------
        Compute and get the gradients of all the clients (Byzantine include)

        Returns
        --------
        Lists of gradients of all clients (honest clients and byzantine)
        """
        [client.compute_gradients() for client in self.client_list]

        honest_gradients = [
            c.get_flat_gradients() 
            for c in self.client_list
        ]

        if self.labelflipping:
            flipped_gradients = [
                c.get_flat_flipped_gradients() 
                for c in self.client_list
            ]
            byzantine_gradients = self.byz_worker.apply_attack(flipped_gradients)
        else:
            byzantine_gradients = self.byz_worker.apply_attack(honest_gradients)

        return honest_gradients + byzantine_gradients
    
    def get_momentum(self):
        """
        Description
        -----------
        Compute and get the momentums of all the clients (Byzantine include)

        Returns
        --------
        Lists of momentums of all clients (honest clients and byzantine)
        """
        [client.compute_gradients() for client in self.client_list]

        honest_momentum = [
            c.get_flat_gradients_with_momentum() 
            for c in self.client_list
        ]
       
        if self.labelflipping:
            flipped_gradients = [
                c.get_flat_flipped_gradients() 
                for c in self.client_list
            ]
            byzantine_gradients = self.byz_worker.apply_attack(flipped_gradients)
        else:
            byzantine_gradients = self.byz_worker.apply_attack(honest_momentum)

        return honest_momentum + byzantine_gradients
    
    def get_batch_norm_stats(self):
        """
        Description
        -----------
        Get the bath-norm stats of all the honest clients

        Returns
        --------
        Lists of bath-norm stats of all honest clients
        """
        batch_norm_stats = [client.get_flat_batch_norm_stats() for client in self.client_list]
        running_mean_stats = [batch_norm_stats_client[0] for batch_norm_stats_client in batch_norm_stats]
        running_var_stats = [batch_norm_stats_client[1] for batch_norm_stats_client in batch_norm_stats]
        byzantine_running_mean = self.byz_worker.apply_batch_norm_attack(running_mean_stats)
        byzantine_running_var = self.byz_worker.apply_batch_norm_attack(running_var_stats)
        return running_mean_stats + byzantine_running_mean, running_var_stats + byzantine_running_var

    
    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Sets the state_dict as the state dict of all the clients.

        Parameters
        ----------
        state_dict(dictionary): state_dict that is desired to be
        set on all the clients.
        """
        [
            client.set_model_state(state_dict) 
            for client in self.client_list
        ]
    
    def transmit_parameters_to_clients(self, parameters):
        """
        Description
        -----------
        Sets the parameters as the parameters of the model of all the clients.

        Parameters
        ----------
        parameters (list): flat vector with the parameters desired to be
        set on all clients.
        """
        [
            client.set_parameters(parameters) 
            for client in self.client_list
        ]

    def get_loss_list_of_clients(self):
        """
        Description
        ------------
        Get the loss list of all clients

        Returns
        -------
        Matrix with the losses that have been computed over the training
        for every client.
        """
        return [client.get_loss_list() for client in self.client_list]
    
    def compute_batch_norm_keys(self):
        """
        Description
        -----------
        Compute batch normalization keys for each client.

        This function iterates over all clients in the Compute Cluster and 
        computes batch normalization keys for each one.

        Returns
        -------
        None
        """
        [client.compute_batch_norm_keys() for client in self.client_list]
    
    def get_train_acc_of_clients(self):
        """
        Description
        -----------
        Retrieve the training accuracy of all clients.

        Returns
        -------
        List
            A list containing the training accuracy for all clients in the Compute Cluster
        """
        return [client.get_train_accuracy() for client in self.client_list]
