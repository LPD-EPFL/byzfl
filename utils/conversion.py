import collections

import torch

def flatten_dict(dict):
    flatten_vector = []
    for _, value in dict.items():
        flatten_vector.append(value.view(-1))
    if len(flatten_vector) > 0:
        return torch.cat(flatten_vector).view(-1)
    else:
        return torch.Tensor(flatten_vector)

def flatten_generator(generator):
    flatten_vector = []
    for item in generator:
        flatten_vector.append(item[1].view(-1))
    if len(flatten_vector) > 0:
        return torch.cat(flatten_vector).view(-1)
    else:
        return torch.Tensor(flatten_vector)

def unflatten_dict(state_dict, flat_vector):
        new_dict = collections.OrderedDict()
        c = 0
        for key, value in state_dict.items():
            nb_elements = torch.numel(value) 
            new_dict[key] = flat_vector[c:c+nb_elements].view(value.shape)
            c = c + nb_elements
        return new_dict

def unflatten_generator(generator, flat_vector):
    new_dict = collections.OrderedDict()
    c = 0
    for item in generator:
        key = item[0]
        nb_elements = torch.numel(item[1]) 
        new_dict[key] = flat_vector[c:c+nb_elements].view(item[1].shape)
        c = c + nb_elements
    return new_dict