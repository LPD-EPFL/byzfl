import torch, random
from robust_aggregators import RobustAggregator
import models, misc
import sys
import collections

class Client(object):
    def __init__(self, 
                 training_dataloader,
                 test_dataloader,
                 model,
                 weight_decay = 1e-4, 
                 loss,
                 lr = 0.1, 
                 lr_decay = None, 
                 momentum = None,
                 milestones = None,
                 labelflipping = False, 
                 device = 'cpu'):

        self.training_dataloader = training_dataloader
        self.test_dataloader = test_dataloader

        self.model = getattr(models, model)().to(device)

        self.lr = lr        
        self.criterion = getattr(torch.nn, loss)()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr = lr, 
                                         weight_decay = weight_decay, 
                                         momentum = momentum, 
                                         dampening = 1 - momentum)

        self.scheduler = torch.optim.scheduler.MuliStepLR(self.optimizer, 
                                          milestones = milestones, 
                                          gamma = lr_decay)

        self.gradient_LF = 0
        self.labelflipping = labelflipping



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


    def compute_gradients(self):

        #JS: compute gradient on flipped labels
        if self.labelflipping:
            self.model.eval()
            self.model.zero_grad()
            targets_flipped = targets.sub(self.numb_labels - 1).mul(-1)
            outputs = sel.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.gradient_LF = self.get_gradients()

        self.model.train()
        self.model.zero_grad()
        inputs, targets = self.sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = sel.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

    def set_parameters(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        self.model.load_state_dict(new_dict)

    def get_gradients(self):
        new_dict = collections.OrderedDict()
        for key, value in self.model.named_parameters():
            new_dict[key] = value.grad
        return new_dict

    def get_parameters(self):
        return self.model.state_dict()

    def step(step):
        self.optimizer.step()
        self.scheduler.step()
