import byzfl.attacks.attacks as attacks
import byzfl.attacks.attack_optimizers as attack_optimizers

class ByzantineWorker():
    """
    Description
    -----------
    Class that is responsible for carrying out the byzantine attacks.

    Parameters
    ----------
        All this parameters should be passed in a dictionary that contains the following keys.
    nb_byz : int
        Number of byzantine nodes that has the byzantine worker.
    attack_name : str
        Name of the attack
    attack_parameters : dict
        Dictionary with the parameters of the attack 
        where every key is the name of the paramater and their value is 
        the value of the parameter.
    attack_optimizer_name : (str, optional)
        Name of the optimizer to be used to find the best attack parameters.
    attack_optimizer_parameters : (dict, optional)
        Dictionary with the parameters of the optimizer where every key is 
        the name of the paramater and their value is the value of the parameter.
    aggregator_info : dict
        Dictionary with the keys "name" and "parameters" defined for the aggregator.
    pre_agg_list : list 
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    

    Methods
    -------
    """
    def __init__(self, params):
        self.nb_real_byz = params["nb_byz"]
        self.attack = getattr(
            attacks, 
            params["attack_name"]
        )(**params["attack_parameters"])

        self.optimizer = None
        self.optimizer_batch_norm = None
        
        if params["attack_optimizer_name"] is not None:
            optimizer_params = {
                "agg_info": params["aggregator_info"],
                "pre_agg_list": params["pre_agg_list"],
                "nb_byz": params["nb_byz"],
                **params["attack_optimizer_parameters"]
            }
            self.optimizer = getattr(
                attack_optimizers, 
                params["attack_optimizer_name"]
            )(**optimizer_params)

            self.optimizer_batch_norm = getattr(
                attack_optimizers, 
                params["attack_optimizer_name"]
            )(**optimizer_params)
    
    def apply_attack(self, honest_vectors):
        """
        Computes the byzantine vector and (optimized or not depending
        if the optimizer is configured) and then it returns a list
        with this vector n times, where n = number of byzantine nodes

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) where each row represents a vector.
        
        Returns
        -------
        Returns a list with the byzantine vector n times, 
        where n = number of byzantine nodes
        """
        if self.nb_real_byz == 0:
            return list()
        if self.optimizer is not None:
            self.optimizer(self.attack, honest_vectors)
        byz_vector = self.attack(honest_vectors)
        
        return [byz_vector] * self.nb_real_byz
    
    def apply_batch_norm_attack(self, honest_vectors):
        """
        Computes the byzantine vector of the batch norms 
        and (optimized or not depending if the optimizer is configured) 
        and then it returns a list with this vector n times, where n = number 
        of byzantine nodes

        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A list of vectors or a matrix (2D array/tensor) where each row represents a vector.
        
        Returns
        -------
        Returns a list with the byzantine vector n times, 
        where n = number of byzantine nodes
        """
        if self.nb_real_byz == 0:
            return list()
        if self.optimizer_batch_norm is not None:
            self.optimizer_batch_norm(self.attack, honest_vectors)
        byz_vector = self.attack(honest_vectors)
        
        return [byz_vector] * self.nb_real_byz
