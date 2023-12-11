import torch, random
from robust_aggregators import RobustAggregator
import models, misc
import sys
import collections

class Client(object):
    def __init__(self, 
                 model,
                 momentum = None,
                 weight_decay = 1e-4, 
                 lr = 0.1, 
                 lr_decay = None, 
                 milestones = None, 
                 device = 'cpu'):


        self.model = getattr(models, model)().to(device).eval()

        self.lr = lr        
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr = lr, 
                                         weight_decay = weight_decay)

        self.scheduler = torch.optim.scheduler.MuliStepLR(self.optimizer, 
                                          milestones = milestones, 
                                          gamma = lr_decay)


    def flatten(state_dict):
        flatten_vector = []
        for key, value in state_dict.items():
            flatten_vector.append(value.view(-1))
        return torch.cat(flatten_vector).view(-1)
 
    def unflatten_parameters(flat_vector):
        new_dict = collections.OrderedDict()
        c = 0
        for key, value in self.model.state_dict():
            nb_elements = torch.numel(value) 
            new_dict[key] = flat_vector[c:c+nb_elements].view(value.shape)
            c = c + nb_element
        return new_dict

    def set_parameters(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        self.model.load_state_dict(new_dict)

    def get_gradients(self):
        new_dict = collections.OrderedDict()
        for key, value in self.model.named_parameters():
            new_dict[key] = value
        return new_dict

    def get_parameters(self):
        return self.model.state_dict()

    def step(step):
        self.optimizer.step()
        self.scheduler.step()
