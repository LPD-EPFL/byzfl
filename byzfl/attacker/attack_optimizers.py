from byzfl.utils.misc import check_vectors_type, random_tool
from byzfl.pipeline.robust_aggregators import RobustAggregator

class LineMaximize():
    """
    Description
    -----------
    Class to optimize attacks using the Line Maximize method: Best-effort arg-maximize a function: ℝ⁺⟶ ℝ, by mere exploration.

    Parameters
    ----------
    agg_info : dict 
        Dictionary with the keys "name" and "parameters" defined.
    pre_agg_info : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.
    nb_byz : int
        Number ob byzantine nodes
    evals : int
        Maximum number of evaluations, must be a positive integer
    start : float
        Initial x evaluated, must be a non-negative float
    delta : float
        Initial step delta, must be a positive float
    ratio : float
        Contraction ratio, must be between 0.5 and 1. (both excluded)

    How to use it in experiments
    ----------------------------
    >>> "attack_optimizer": {
    >>>     "name": "LineMaximize",
    >>>     "parameters": {
    >>>         "evals": 16,
    >>>         "start": 0,
    >>>         "delta": 1,
    >>>         "ratio": 0.8
    >>>     }
    >>> }

    Methods
    -------                        
    """
    def __init__(self, agg_info, pre_agg_list=[], nb_byz=0, evals=16, start=0., delta=1., ratio=0.8):
        self.robust_aggregator = RobustAggregator(
            agg_info,
            pre_agg_list
        )
        
        self.nb_byz = nb_byz
        self.evals = evals
        self.start = start
        self.delta = delta
        self.ratio = ratio
    
    def _evaluate(self,
                  attack,
                  honest_vectors,
                  avg_honest_vector,
                  attack_factor):
        """
        Compute the norm of the distance beetwen
        the difference of the honest vectors with the byzantine vectors
        aggregated against the average of the honest vectors.

        Parameters
        -----------
        attack : Attack 
            Class that simulates the attack we would like to reproduce.
            It is important that this class has the set_attack_parameters and
            get_malicious_vectors implemented. (See Fall of Empires or
            LittleIsEnough from attacks.py)

        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Vectors from the honest nodes

        avg_honest_vector : 1D ndarray or 1D torch.tensor with floating point or complex dtype
            Average from the honest_vectors. 
            (Note: Could be computed inside but for questions of efficiency it's passed by parameter)
        
        attack_factor : (float)
            Attack factor to set in the attack

        Returns
        -------
        The Norm of the distance beetwen the difference of the honest vectors
        with the byzantine vectors aggregated against the average of the 
        honest vectors.
        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        
        attack.set_attack_parameters(attack_factor)

        byzantine_vector = attack(honest_vectors)

        byzantine_vectors = tools.array([byzantine_vector] * self.nb_byz)

        vectors = tools.concatenate((honest_vectors, byzantine_vectors), axis=0)
        
        agg_vectors = self.robust_aggregator(vectors)
        
        distance = tools.subtract(agg_vectors, avg_honest_vector)
        
        return tools.linalg.norm(distance)
    
    def __call__(self, attack, honest_vectors):
        """
        Iterative algorithm to set the
        best attack factor to the attack
        given the attributs of the class.

        Parameters
        -----------
        attack : Attack
            Class that simulates the attack we would like to reproduce.
            It is important that this class has the set_attack_parameters and
            get_malicious_vectors implemented. (See Fall of Empires or
            LittleIsEnough from attacks.py)

        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Vectors from the honest nodes

        Note
        ----
        This function doesn't return anything but it sets to the attack
        passed by parameter the best attack factor.
        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        # Variable setup
        evals = self.evals
        delta = self.delta
        avg_honest_vector = tools.mean(honest_vectors, axis=0)
        best_x = self.start
        best_y = self._evaluate(attack, honest_vectors,
                                avg_honest_vector, best_x)
        evals -= 1
        # Expansion phase
        while evals > 0:
            prop_x = best_x + delta
            prop_y = self._evaluate(attack, honest_vectors,
                                    avg_honest_vector, prop_x)
            evals -= 1
            # Check if best
            if prop_y > best_y:
                best_y = prop_y
                best_x = prop_x
                delta *= 2
            else:
                delta *= self.ratio
                break
        # Contraction phase
        while evals > 0:
            if prop_x < best_x:
                prop_x += delta
            else:
                x = prop_x - delta
                while x < 0:
                    x = (x + prop_x) / 2
                prop_x = x
            # Same input in old doesn't correspond to same output
            # With same factor
            prop_y = self._evaluate(attack, honest_vectors,
                                    avg_honest_vector, prop_x)
            evals -= 1
            # Check if best
            if prop_y > best_y:
                best_y = prop_y
                best_x = prop_x
            # Reduce delta
            delta *= self.ratio
        # Return found maximizer
        attack.set_attack_parameters(best_x)


class WorkerWithMaxVariance():
    """
    Description
    -----------
    Class to optimize attacks that need to focus on the worker
    with maximum variance. This clase evaluates the honest vectors
    and sets the attack to focus on the worker with maximum variance.

    How to use it in experiments
    ----------------------------
    >>> "attack_optimizer": {
    >>>     "name": "WorkerWithMaxVariance",
    >>>     "parameters": {
    >>>         "steps_to_learn": 20
    >>>     }
    >>> }

    Parameters
    ----------
    steps_to_learn : int
        How many steps we are computing which worker to mimic.

    Methods
    -------          
    """
    def __init__(self, steps_to_learn=None, **kwargs):
        self.z = None
        self.mu = None
        self.steps_to_learn = steps_to_learn
        self.current_step = -1
    
    def _update_heuristic(self, honest_vectors):
        """
        Private function used to compute and update the atributs of their
        heuristic every round for computing the best z.

        Parameters
        -----------
        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Vectors from the honest nodes
        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        
        if self.z is None:
            rand_tool = random_tool(honest_vectors)
            self.z = rand_tool.rand(honest_vectors[0])
        if self.mu is None:
            self.mu = tools.zeros_like(honest_vectors[0])
        
        time_factor = 1 / (self.current_step + 1)
        step_ratio = (self.current_step) * time_factor
        self.mu = tools.multiply(self.mu, step_ratio)
        self.mu = tools.add(
            self.mu,
            tools.multiply(tools.mean(honest_vectors, axis=0), time_factor)
        )

        deviations = tools.subtract(honest_vectors, self.mu)
        dot_product = tools.dot(deviations, self.z)
        dev_scaled = tools.multiply(deviations, dot_product[:,None])
        
        cumulative = tools.sum(dev_scaled, axis=0)
        cumulative = tools.divide(cumulative, tools.linalg.norm(cumulative))

        self.z = tools.multiply(self.z, step_ratio)
        self.z = tools.add(self.z, tools.multiply(cumulative, time_factor))
        self.z = tools.divide(self.z, tools.linalg.norm(self.z))
    
    def __call__(self, attack, honest_vectors):
        """
        Optimize the attack by setting their parameter to the ID 
        of the worker with more variance

        Parameters
        -----------
        attack : Attack 
            Class that simulates the attack we would like to reproduce.
            It is important that this class has the set_attack_parameters and
            get_malicious_vectors implemented.

        honest_vectors : 2D ndarray or 2D torch.tensor with floating point or complex dtype
            Vectors from the honest nodes
                
        Note
        ----
        This function doesn't return anything but it sets to the attack
        passed by parameter the ID of the worker with maximum variance.

        """
        tools, honest_vectors = check_vectors_type(honest_vectors)
        self.current_step += 1
        parameter = None

        if self.steps_to_learn is None:
            parameter = 0
        elif self.current_step < self.steps_to_learn:
            parameter = 0
            self._update_heuristic(honest_vectors)
        else:
            dot_products = abs(tools.dot(honest_vectors, self.z))
            parameter = tools.argmax(dot_products)

        attack.set_attack_parameters(parameter)

