import random
import time

import numpy as np
import torch

from byzfl.pipeline.server import Server
from byzfl.pipeline.compute_cluster import ComputeCluster
from byzfl.pipeline.dataset import get_dataloaders
from byzfl.pipeline.managers import FileManager, ParamsManager

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
        Batch size used in the validation and test dataloader
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
    distribution_parameter : float
        Parameter for the data distribution
    aggregator_info : dict
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_list : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    nb_workers : int
        Number of workers
    nb_byz : int
        Number of byzantine nodes
    declared_nb_byz : int
        Number of byzantine nodes that the server will try to defend against
    loss_name : str 
        Loss name to be used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning_rate_decay : float
        Rate decreases over time during training
    momentum : float
        Momentum
    batch_norm_momentum : float
        Momentum used in the federated batch norm
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
    store_models : (bool)
        If true every delta steps the model will be saved

    Methods
    -------
    """
    def __init__(self, raw_params):

        params_manager = ParamsManager(params=raw_params)
        params = params_manager.get_flatten_info()

        if params["nb_honest"] is not None:
            params["nb_workers"] = params["nb_honest"] + params["declared_nb_byz"]

        file_manager_params = {
            "result_path": params["results_directory"],
            "dataset_name": params["dataset_name"],
            "model_name": params["model_name"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"],
            "declared_nb_byz": params["declared_nb_byz"],
            "data_distribution_name": params["data_distribution_name"],
            "distribution_parameter": params["distribution_parameter"],
            "aggregation_name": params["aggregator_info"]["name"],
            "pre_aggregation_names":  [
                dict['name']
                for dict in params["pre_agg_list"]
            ],
            "attack_name": params["attack_name"],
            "learning_rate": params["learning_rate"],
            "momentum": params["momentum"],
            "weight_decay": params["weight_decay"],
            "learning_rate_decay": params["learning_rate_decay"]
        }

        self.file_manager = FileManager(file_manager_params)
        self.file_manager.save_config_dict(params_manager.get_data())

        params["aggregator_info"]["parameters"]["f"] = params["declared_nb_byz"]
        if len(params["pre_agg_list"]) > 0:
            for pre_agg in params["pre_agg_list"]:
                pre_agg["parameters"]["f"] = params["declared_nb_byz"]

        params_dataloaders = {
            "dataset_name": params["dataset_name"],
            "batch_size": params["batch_size"],
            "batch_size_validation": params["batch_size_validation"],
            "size_train_set": params["size_train_set"],
            "dataset_name": params["dataset_name"],
            "nb_labels": params["nb_labels"],
            "data_folder": params["data_folder"],
            "data_distribution_name": params["data_distribution_name"],
            "distribution_parameter": params["distribution_parameter"],
            "nb_workers": params["nb_workers"],
            "nb_byz": params["nb_byz"],
        }

        self.data_distribution_seed = params["data_distribution_seed"]
        
        # Deterministic
        # https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(self.data_distribution_seed)
        torch.manual_seed(self.data_distribution_seed)
        torch.use_deterministic_algorithms(True)
        random.seed(self.data_distribution_seed)

        train_dataloaders, self.validation_dataloader, \
        self.test_dataloader = get_dataloaders(params_dataloaders)

        self.use_validation = self.validation_dataloader != None

        self.training_seed = params["training_seed"]
        np.random.seed(self.training_seed)
        torch.manual_seed(self.training_seed)
        random.seed(self.training_seed)

        server_params = {
            "model_name": params["model_name"],
            "nb_workers": params["nb_workers"],
            "evaluate_on_test": params["evaluate_on_test"],
            "device": params["device"],
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "milestones": params["milestones"],
            "learning_rate_decay": params["learning_rate_decay"],
            "aggregator_info": params["aggregator_info"],
            "pre_agg_list": params["pre_agg_list"],
            "batch_size": params["batch_size"],
            "batch_norm_momentum": params["batch_norm_momentum"]
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
            "nb_labels": params["nb_labels"],
            "nb_steps": params["nb_steps"]
        }

        self.compute_cluster = ComputeCluster(compute_cluster_params)

        self.compute_cluster.set_model_state(self.server.get_dict_parameters())

        self.device = params["device"]
        self.steps = params["nb_steps"]
        self.evaluation_delta = params["evaluation_delta"]
        self.evaluate_on_test = params["evaluate_on_test"]
        self.store_training_accuracy = params["store_training_accuracy"]
        self.store_training_loss = params["store_training_loss"]
        self.store_models = params["store_models"]

        #Stored for display results only
        self.agg_name = params["aggregator_info"]["name"]
        self.data_dist_name = params["data_distribution_name"]
        self.attack_name = params["attack_name"]
        self.nb_byz = params["nb_byz"]

        self.accuracy_list = np.array([])
        self.test_accuracy_list = np.array([])
        self.loss_list = np.array([])

        self.use_batch_norm_stats = False
    
    @torch.no_grad()
    def _compute_accuracy(self, model, dataloader):
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
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct/total

    def run_SGD(self):
        """
        Description
        -----------
        Trains the model running SGD for n steps (setting.json)
        """
        start_time = time.time()

        self.server.compute_batch_norm_keys()

        if self.server.use_batch_norm():
            self.use_batch_norm_stats = True
            self.compute_cluster.compute_batch_norm_keys()
        
        for step in range(0, self.steps):

            if step % self.evaluation_delta == 0:
                model = self.server.get_model()
                if self.use_validation:
                    accuracy = self._compute_accuracy(model, self.validation_dataloader)
                    self.accuracy_list = np.append(self.accuracy_list, accuracy)

                if self.evaluate_on_test:
                    test_accuracy = self._compute_accuracy(model, self.test_dataloader)
                    self.test_accuracy_list = np.append(
                        self.test_accuracy_list, 
                        test_accuracy
                    )
                    self.file_manager.write_array_in_file(
                        self.test_accuracy_list, 
                        "test_accuracy_tr_seed_" + str(self.training_seed) 
                        + "_dd_seed_" + str(self.data_distribution_seed) +".txt"
                    )
                
                if self.store_models:
                    self.file_manager.save_state_dict(
                        self.server.get_dict_parameters(),
                        self.training_seed,
                        step
                    )
            
            #Training
            new_gradients = self.compute_cluster.get_momentum()

            if self.use_batch_norm_stats:
                new_running_mean, new_running_var = self.compute_cluster.get_batch_norm_stats()

            #Aggregation and update of the global model
            self.server.update_model(new_gradients)

            if self.use_batch_norm_stats:
                self.server.update_batch_norm_stats(new_running_mean, new_running_var)

            #Broadcasting
            new_parameters = self.server.get_dict_parameters()
            self.compute_cluster.set_model_state(new_parameters)
        

        self.loss_list = self.compute_cluster.get_loss_list_of_clients()
        self.train_accuracy_list = self.compute_cluster.get_train_acc_of_clients()

        if self.use_validation:
            accuracy = self._compute_accuracy(model, self.validation_dataloader)
            self.accuracy_list = np.append(self.accuracy_list, accuracy)
            self.file_manager.write_array_in_file(
                self.accuracy_list, 
                "validation_accuracy_tr_seed_" + str(self.training_seed) 
                + "_dd_seed_" + str(self.data_distribution_seed) +".txt"
            )
        
        if self.evaluate_on_test:
            test_accuracy = self._compute_accuracy(model, self.test_dataloader)
            self.test_accuracy_list = np.append(self.test_accuracy_list, test_accuracy)
            self.file_manager.write_array_in_file(
                self.test_accuracy_list, 
                "test_accuracy_tr_seed_" + str(self.training_seed) 
                + "_dd_seed_" + str(self.data_distribution_seed) +".txt"
            )

        if self.store_training_loss:
            for i, loss in enumerate(self.loss_list):
                self.file_manager.save_loss(
                    loss, 
                    self.training_seed, 
                    self.data_distribution_seed, 
                    i
                )
        
        if self.store_training_accuracy:
            for i, acc in enumerate(self.train_accuracy_list):
                self.file_manager.save_accuracy(
                    acc, 
                    self.training_seed, 
                    self.data_distribution_seed,
                    i
                )
        
        if self.store_models:
            self.file_manager.save_state_dict(
                self.server.get_dict_parameters(),
                self.training_seed,
                self.steps
            )
        
        end_time = time.time()
        execution_time = end_time - start_time

        self.file_manager.write_array_in_file(
            np.array(execution_time),
            "train_time_tr_seed_" + str(self.training_seed) 
            + "_dd_seed_" + str(self.data_distribution_seed) +".txt"
        )
