import numpy as np
import torch
import utils.torch_tools as torch_tools

def check_vectors(vectors):
    if ((isinstance(vectors, list) and isinstance(vectors[0], np.ndarray)) or
        isinstance(vectors, np.ndarray)) :
        tool = np
    elif ((isinstance(vectors, list) and isinstance(vectors[0], torch.Tensor))
        or isinstance(vectors, torch.Tensor)) :
        tool = torch_tools
    else :
        raise TypeError("'vectors' should be a 'list' or 'np.ndarray'"
                        " or 'torch.Tensor")
    return tool