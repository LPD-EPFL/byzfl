import numpy as np
import torch
import utils.torch_tools as torch_tools

def check_vectors(vectors):
    if isinstance(vectors, list) and isinstance(vectors[0], np.ndarray):
        tools = np
        vectors = np.array(vectors)
    elif isinstance(vectors, np.ndarray) :
        tools = np
    elif isinstance(vectors, list) and isinstance(vectors[0], torch.Tensor):
        tools = torch_tools
        vectors = torch.stack(vectors)
    elif isinstance(vectors, torch.Tensor) :
        tools = torch_tools
    else :
        raise TypeError("'vectors' should be a 'list' of"
                        " ('np.ndarray' or 'torch.Tensor') or 'np.ndarray'"
                        " or 'torch.Tensor'")
    return tools, vectors

def random_tool(vectors):
    if (isinstance(vectors, list) and isinstance(vectors[0], np.ndarray) or
        isinstance(vectors, np.ndarray)):
        tools = np.random
    elif (isinstance(vectors, list) and isinstance(vectors[0], torch.Tensor) or
          isinstance(vectors, torch.Tensor) ):
        tools = torch_tools

    return tools


