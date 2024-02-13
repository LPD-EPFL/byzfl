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
                                         weight_decay = weight_decay)

        self.scheduler = torch.optim.scheduler.MuliStepLR(self.optimizer, 
                                          milestones = milestones, 
                                          gamma = lr_decay)

        self.gradient_LF = 0
        self.labelflipping = labelflipping
        
        self.mom = momentum
        self.last_mom = 0


    def flatten_dict(state_dict):
        flatten_vector = []
        for key, value in state_dict.items():
            flatten_vector.append(value.view(-1))
        return torch.cat(flatten_vector).view(-1)

    def flatten_generator(state_dict):
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

    def compute_gradients(self):

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

    def get_flat_gradients_with_momentum(self):
        flat_gradient = self.flatten(self.get_gradients())
        new_momentum = self.mom * self.last_mom + (1-self.mom) * flat_gradient
        self.last_momentum = new_momentum
        return new_momentum

    def get_gradients(self, flat = False):
        if flat == True:
            return self.flatten(self.model.named_parameters())
        return self.get_dict_gradients()

    def get_dict_gradients(self):
        new_dict = collections.OrderedDict()
        for key, value in self.model.named_parameters():
            new_dict[key] = value.grad
        return new_dict

    def get_parameters(self, flat = False):
        if flat == True:
            return self.flatten(self.model.state_dict())
        return self.model.state_dict()

    def set_parameters(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        self.model.load_state_dict(new_dict)

    def set_gradients(self, flat_vector):
        new_dict = unflatten_parameters(flat_vector)
        for key, value in self.model.named_parameters():
            value.grad = new_dict[key].detach().clone()

    def step(step):
        self.optimizer.step()
        self.scheduler.step()
