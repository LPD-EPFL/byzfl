from utils.misc import check_vectors_type, random_tool

class LineMaximize():
    def __init__(self,
                 robust_aggregator,
                 nb_byz=0,
                 evals=16,
                 start=0.,
                 delta=1.,
                 ratio=0.8):
        
        self.robust_aggregator = robust_aggregator
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
        
        tools, honest_vectors = check_vectors_type(honest_vectors)

        attack.set_attack_parameters(attack_factor)

        byzantine_vector = attack.get_malicious_vector(honest_vectors)
        byzantine_vectors = tools.array([byzantine_vector] * self.nb_byz)
        vectors = tools.concatenate((honest_vectors, byzantine_vectors),
                                    axis=0)
        distance = self.robust_aggregator.aggregate(vectors) \
                   - avg_honest_vector
        
        return tools.linalg.norm(distance)
    
    def optimize(self, attack, honest_vectors):
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
    def __init__(self, steps_to_learn=None, z=None, mu=None):
        self.z = z
        self.mu = mu
        self.steps_to_learn = steps_to_learn
        self.current_step = -1
    
    def _update_heuristic(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        
        if self.z is None:
            rand_tool = random_tool(honest_vectors)
            self.z = rand_tool.rand(honest_vectors[0])
        if self.mu is None:
            self.mu = tools.zeros_like(honest_vectors[0])

        time_factor = 1 / (self.current_step + 2)
        step_ratio = (self.current_step + 1) * time_factor
        self.mu *= step_ratio
        self.mu = tools.add(self.mu,
                            tools.mean(honest_vectors, axis=0) * time_factor)
        self.z *= step_ratio
        deviations = tools.subtract(honest_vectors, self.mu)
        dot_product = tools.dot(deviations, self.z)
        dev_scaled = deviations * tools.linalg.norm(dot_product) 
        dev_normalized = dev_scaled / tools.linalg.norm(dev_scaled)
        cumulative = tools.sum(dev_normalized, axis=0)

        self.z *= step_ratio
        self.z += cumulative * time_factor
        self.z /= tools.linalg.norm(self.z)

    
    def optimize(self, attack, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        self.current_step += 1

        if self.steps_to_learn is None:
            return 0
        
        if self.current_step < self.steps_to_learn:
            self._update_heuristic(honest_vectors)
            return 0
        
        dot_products = abs(tools.dot(honest_vectors, self.z))
        attack.set_attack_parameters(tools.argmax(dot_products))