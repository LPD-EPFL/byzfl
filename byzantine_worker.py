class ByzantineWorker():
    def __init__(self, attack, nb_real_byz=0, optimizer=None):
        self.attack = attack
        self.nb_real_byz = nb_real_byz
        self.optimizer = optimizer
    
    def apply_attack(self, honest_vectors):
        if self.nb_real_byz == 0:
            return list()
        if self.optimizer is not None:
            self.optimizer.optimize(self.attack, honest_vectors)
        byz_vector = self.attack.get_malicious_vector(honest_vectors)
        
        return [byz_vector] * self.nb_real_byz
