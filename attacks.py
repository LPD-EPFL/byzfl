import numpy as np

from utils.misc import check_vectors_type

class NoAttack():
    def get_malicious_vector(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return mean_vector


class SignFlipping():
    def get_malicious_vector(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, -1)


class FallOfEmpires():
    def __init__(self, attack_factor=3):
        self.attack_factor = attack_factor
    
    def set_attack_parameters(self, factor):
        self.attack_factor = factor

    def get_malicious_vector(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        mean_vector = tools.mean(honest_vectors, axis=0)
        return tools.multiply(mean_vector, 1 - self.attack_factor)


class LittleIsEnough():
    def __init__(self, attack_factor=1.5):
        self.attack_factor = attack_factor
    
    def set_attack_parameters(self, factor):
        self.attack_factor = factor

    def get_malicious_vector(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        attack_vector = tools.sqrt(tools.var(honest_vectors, axis=0, ddof=1))
        return (tools.mean(honest_vectors, axis=0) +
                attack_vector * self.attack_factor)


class Mimic():
    def __init__(self, worker_to_mimic=0):
        self.worker_to_mimic = worker_to_mimic
    
    def set_attack_parameters(self, worker):
        self.worker_to_mimic = worker
    
    def get_malicious_vector(self, honest_vectors):
        return honest_vectors[self.worker_to_mimic]


class Inf():
    def get_malicious_vector(self, honest_vectors):
        tools, honest_vectors = check_vectors_type(honest_vectors)
        return tools.full_like(honest_vectors[0], np.inf)