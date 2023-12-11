import torch, random
from robust_aggregators import RobustAggregator
import models, misc
import sys
import collections

class Server(object):
    def __init__(self, 
                 model,
                 agg_name = "average", 
                 pre_agg_name = None, 
                 bucket_size = None, 
                 nb_byz = 0, 
                 weight_decay = 1e-4, 
                 lr = 0.1, 
                 lr_decay = None, 
                 milestones = None, 
                 device = 'cpu'):


        self.model = getattr(models, model)().to(device).eval()
        self.agg = RobustAggregator(agg_name, 
                                    pre_aggregator = pre_agg_name,
                                    nb_byz = nb_byz,
                                    bucket_size = bucket_size)

        self.lr = lr        
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr = lr, 
                                         weight_decay = weight_decay)

        self.scheduler = torch.optim.scheduler.MultiStepLR(self.optimizer, 
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

    def unflatten_gradients(flat_vector):
        new_dict = collections.OrderedDict()
        c = 0
        for key, value in self.model.named_parameters():
            nb_elements = torch.numel(value) 
            new_dict[key] = flat_vector[c:c+nb_elements].view(value.shape)
            c = c + nb_element
        return new_dict

    def aggregate(self, vectors):
        return self.robust_aggregator.aggregate(vectors)

    def set_parameters(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        self.model.load_state_dict(new_dict)

    def set_gradients(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        for key, value in self.model.named_parameters():
            value.grad = new_dict[key].detach().clone()

    def get_parameters(self):
        return self.model.state_dict()

    def step(step):
        self.optimizer.step()
        self.scheduler.step()
