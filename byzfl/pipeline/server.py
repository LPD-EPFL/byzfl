from byzfl.pipeline.robust_aggregators import RobustAggregator
from byzfl.pipeline.model_base_interface import ModelBaseInterface
from byzfl.utils.conversion import unflatten_dict

class Server(ModelBaseInterface):
    """
    Description
    -----------
    This class simulates the central server of our environment
    where the global model is updated.

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    model_name : str
        Indicates the model to be used
    dataloader : Dataloader
        Dataloader with the validation set to compute the accuracy of the global model
    test_dataloader : Dataloader
        Dataloader with the validation set to compute the test accuracy of the global model
    device : str 
        Name of the device used
    bit_precision : int
        How many bits will be displayed in the accuracy
    learning_rate : float 
        Learning rate
    weight_decay : float 
        Weight decay used
    milestones : list 
        List of the milestones, where the learning rate decay should be applied
    learning-rate-decay : float
        Lerning rate decay used
    dataset_name : str 
        Name of the dataset used
    aggregator_info : dict 
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_info : list
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
            "learning_rate_decay": params["learning_rate_decay"],
            "nb_workers": params["nb_workers"]
        })
        self.robust_aggregator = RobustAggregator(
            params["aggregator_info"],
            params["pre_agg_list"]
        )

        # Clipping should not be applied to batch norm stats
        batch_norm_preagg_list = []
        for pre_agg in params["pre_agg_list"]:
            if pre_agg["name"] != "Clipping":
                batch_norm_preagg_list.append(pre_agg)

        # Bath Norm Robust Aggregators
        self.robust_aggregator_mean = RobustAggregator(
            params["aggregator_info"],
            batch_norm_preagg_list
        )
        self.robust_aggregator_var = RobustAggregator(
            params["aggregator_info"],
            batch_norm_preagg_list
        )
        self.robust_aggregator_bias = RobustAggregator(
            params["aggregator_info"],
            batch_norm_preagg_list
        )

        self.model.eval()

        #Needed for batch norm
        self.batch_size = params["batch_size"]
        self.nb_workers = params["nb_workers"]
        self.batch_norm_momentum = params["batch_norm_momentum"]

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
        return self.robust_aggregator(vectors)
    
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
    
    def step(self):
        """
        Description
        -----------
        Do a step of the optimizer and the scheduler.
        """
        self.optimizer.step()
        self.scheduler.step()
    
    def get_model(self):
        return self.model
    
    def compute_validation_accuracy(self):
        return self._compute_accuracy(self.validation_loader)
    
    def compute_test_accuracy(self):
        return self._compute_accuracy(self.test_loader)
    
    def update_batch_norm_stats(self, new_running_mean, new_running_var):
        """
        Description
        -----------
        Update the model aggregating the bath norm statistics given.

        Parameters
        ----------
        batch_norm_stats : list
            Flat list with the bath norm statistics
        """
        agg_mean = self.robust_aggregator_mean(new_running_mean)
        agg_var = self.robust_aggregator_var(new_running_var)

        if self.use_federated_batch_norm():
            list_of_bias = [(param - agg_mean)**2 for param in new_running_mean]
            agg_bias = self.robust_aggregator_bias(list_of_bias)
            agg_var = agg_var + (self.nb_workers*self.batch_size / (self.batch_norm_momentum * (self.nb_workers*self.batch_size - 1))) * agg_bias
        
        self.set_batch_norm_stats(agg_mean, agg_var)

    def set_batch_norm_stats(self, agg_running_mean, agg_running_var):
        """
        Description
        -----------
        Sets the model batch norm stats given a flat vector.

        Parameters
        ----------
        flat_vector : list
            Flat list with the parameters
        """
        dictionary_mean, dictionary_var = self.get_batch_norm_stats()
        state_dict = self.model.state_dict()
        agg_mean_stats = unflatten_dict(dictionary_mean, agg_running_mean)
        for key, item in agg_mean_stats.items():
            state_dict[key] = item

        agg_var_stats = unflatten_dict(dictionary_var, agg_running_var)
        for key, item in agg_var_stats.items():
            state_dict[key] = item

        self.set_model_state(state_dict)
