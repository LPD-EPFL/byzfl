import numpy as np
import torch
from scipy.spatial import distance
from byzfl.utils import torch_tools
import os, random

def check_vectors_type(vectors):
    if isinstance(vectors, list) and isinstance(vectors[0], np.ndarray):
        tools = np
        vectors = np.array(vectors)
    elif isinstance(vectors, np.ndarray):
        tools = np
    elif isinstance(vectors, list) and isinstance(vectors[0], torch.Tensor):
        tools = torch_tools
        vectors = torch.stack(vectors).float()
    elif isinstance(vectors, torch.Tensor):
        tools = torch_tools
        vectors = vectors.float()
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
          isinstance(vectors, torch.Tensor)):
        tools = torch_tools
    return tools


def distance_tool(vectors):
    if (isinstance(vectors, list) and isinstance(vectors[0], np.ndarray) or
        isinstance(vectors, np.ndarray)):
        tools = distance
    elif (isinstance(vectors, list) and isinstance(vectors[0], torch.Tensor) or
          isinstance(vectors, torch.Tensor)):
        tools = torch_tools
    return tools


def set_random_seed(seed: int):
    """
    Set the random seed for reproducibility across different libraries and environments.

    Parameters
    ----------
    seed : int
        The seed value to use for random number generation.
    """
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)