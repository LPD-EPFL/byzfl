import numpy as np

from server import Server
from compute_cluster import ComputeCluster
from dataset import get_dataloaders

class Train(object):
    """
    Description
    -----------
    Class with implements the train algorithms and stores some stadistics
    of the training

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    nb_steps : int 
        Number of steps to do in the Training
    evaluation_delta : int
        How many steps it have to wait to compute accuracy
    evaulate_on_test : bool
        Indicates if will be evaluate on test set
    model_name : str
        Indicates the model to be used
    device : str 
        Name of the device used
    bit_precision : int
        How many bits will be displayed in the accuracy
    learning_rate : float
        Learning rate
    weight_decay : float 
        Weight decay used
    learning_rate_decay : float
        Lerning rate decay used
    batch_size : int
        Batch size used in the train dataloaders
    batch_size_validation: int
        Batch size used in the validation dataloader
    size_train_set: float
        Proportion of data from the train that will be used to train
    dataset_name : str 
        Name of the dataset used
    nb_labels : int
        Number of labels in the dataset
    data_folder : str
        Path of the folder that contains the data
    data_distribution_name : str
        Name of the data distribution
    data_distribution_parameters : dict
        Parameters of the data distribution
    aggregator_info : dict
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_list : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    nb_workers : int
        Number of workers
    nb_byz : int
        Number of byzantine nodes
    loss_name : str 
        Loss name to be used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning_rate_decay : float
        Rate decreases over time during training
    momentum : float
        Momentum
    attack_name : str 
        Attack name
    attack_parameters : dict 
        Dictionary with the parameters of the attack where every key is the name 
        of the paramater and their value is the value of the parameter
    attack_optimizer_name : (str, optional)
        Name of the optimizer to be used
    attack_optimizer_parameters : (dict, optional)
        Dictionary with the parameters of the optimizer where every 
        key is the name of the paramater and their value is the value 
        of the parameter

    Methods
    -------
    """
    def __init__(self, params):
        params["aggregator_info"]["parameters"]["nb_byz"] = params["nb_byz"]
        if len(params["pre_agg_list"]) > 0:
            for pre_agg in params["pre_agg_list"]:
                pre_agg["parameters"]["nb_byz"] = params["nb_byz"]
        
        if params["nb_honest"] is not None:
            params["nb_workers"] = params["nb_honest"] + params["nb_byz"]

        params_dataloaders = {
            "dataset_name": params["dataset_name"],
            "batch_size": params["batch_size"],
            "batch_size_validation": params["batch_size_validation"],
            "size_train_set": params["size_train_set"],
            "dataset_name": params["dataset_name"],
            "nb_labels": params["nb_labels"],
            "data_folder": params["data_folder"],
            "data_distribution_name": params["data_distribution_name"],
            "data_distribution_parameters": params["data_distribution_parameters"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"]
        }

        train_dataloaders, validation_dataloader, test_dataloader = get_dataloaders(params_dataloaders)

        server_params = {
            "model_name": params["model_name"],
            "dataloader": validation_dataloader,
            "test_dataloader": test_dataloader,
            "evaluate_on_test": params["evaluate_on_test"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"],
            "bit_precision": params["bit_precision"]
        }

        self.server = Server(server_params)

        if params["nb_byz"] == 0:
            params["attack_name"] = "NoAttack"

        compute_cluster_params = {
            "device": params["device"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"],
            "model_name": params["model_name"],
            "learning_rate": params["learning_rate"],
            "loss_name": params["loss_name"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "momentum": params["momentum"],
            "attack_name": params["attack_name"],
            "attack_parameters": params["attack_parameters"],
            "attack_optimizer_name": params["attack_optimizer_name"],
            "attack_optimizer_parameters": params["attack_optimizer_parameters"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"],
            "dataloaders": train_dataloaders,
            "nb_labels": params["nb_labels"]
        }

        self.compute_cluster = ComputeCluster(compute_cluster_params)

        self.compute_cluster.set_model_state(self.server.get_dict_parameters())

        self.steps = params["nb_steps"]
        self.evaluation_delta = params["evaluation_delta"]
        self.evaluate_on_test = params["evaluate_on_test"]
        self.accuracy_list = np.array([])
        self.test_accuracy_list = np.array([])
        self.model_list = np.array([])
        self.loss_list = -1
    
    def get_accuracy_list(self):
        """
        Description
        -----------
        Get accuracy list

        Returns
        -------
        List of accuracies
        """
        return self.accuracy_list

    def get_test_accuracy_list(self):
        """
        Description
        -----------
        Get test accuracy list

        Returns
        -------
        List of accuracies
        """
        return self.test_accuracy_list
    
    def get_model_list(self):
        """
        Description
        -----------
        Get a list with the parameters of the models
        that have been evaluated

        Returns
        -------
        List of state_dicts
        """
        return self.model_list
    
    def get_loss_list(self):
        """
        Description
        -----------
        Get a list with the loss of the clients
        computed every step

        Returns
        -------
        Matrix of losses
        """
        return self.loss_list
    
    def run_SGD(self):
        """
        Description
        -----------
        Trains the model running SGD for n steps (setting.json)
        """
        for step in range(0, self.steps):

            if step % self.evaluation_delta == 0:
                accuracy = self.server.compute_validation_accuracy()
                self.accuracy_list = np.append(self.accuracy_list, accuracy)
                self.model_list = np.append(
                    self.model_list, 
                    self.server.get_dict_parameters()
                )

                if self.evaluate_on_test:
                    test_accuracy = self.server.compute_test_accuracy()
                    self.test_accuracy_list = np.append(
                        self.test_accuracy_list, 
                        test_accuracy
                    )
            
            #Training
            new_gradients = self.compute_cluster.get_momentum()
            new_batch_norm_stats = self.compute_cluster.get_batch_norm_stats()

            #Aggregation and update of the global model
            self.server.update_model(new_gradients)
            if new_batch_norm_stats[0].numel() != 0:
                self.server.update_batch_norm_stats(new_batch_norm_stats)

            #Broadcasting
            new_parameters = self.server.get_flat_parameters()
            self.compute_cluster.transmit_parameters_to_clients(new_parameters)
        
        accuracy = self.server.compute_validation_accuracy()
        self.accuracy_list = np.append(self.accuracy_list, accuracy)
        self.model_list = np.append(self.model_list, self.server.get_dict_parameters())
        self.loss_list = self.compute_cluster.get_loss_list_of_clients()

        if self.evaluate_on_test:
            test_accuracy = self.server.compute_test_accuracy()
            self.test_accuracy_list = np.append(self.test_accuracy_list, test_accuracy)
