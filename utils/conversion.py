


def from_dict_to_flat_tensor(state_dict):
    flatten_vector = []
    for key, value in state_dict.items():
        flatten_vector.append(value.view(-1))
    return torch.cat(flatten_vector).view(-1)

def from_generator_to_flat_tensor(generator):
    flatten_vector = []
    for item in generator:
    	if isinstance(item, tuple) and len(item)==2:
    		_, value = item
    	else :
    		value = item
        flatten_vector.append(value.view(-1))
    return torch.cat(flatten_vector).view(-1)