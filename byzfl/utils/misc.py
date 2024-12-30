import numpy as np
import torch
from scipy.spatial import distance
from byzfl.utils import torch_tools


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


def check_type(element, t):
    if isinstance(t, tuple):
        s = "" 
        for i in range(len(t)):
            s = s + "'"+ t[i].__name__+"'"
            if i < len(t)-1:
                s = s + " or "
    else:
        s = "'" + t.__name__+ "'"
    if not isinstance(element, t):
        raise TypeError("Expected type" + s + " but got '" + type(element).__name__ + "'")


#JS
def check_greater_than_value(element, element_name, value):
    if element <= value:
        raise ValueError("Expected value greater than " + str(value) + " for " + element_name)


#JS
def check_greater_than_or_equal_value(element, element_name, value):
    if element < value:
        raise ValueError("Expected value greater than or equal to " + str(value) + " for " + element_name)


#JS
def check_smaller_than_value(element, element_name, value):
    if element >= value:
        raise ValueError("Expected value smaller than " + str(value) + " for " + element_name)
    
#JS
def check_different_from_value(element, element_name, value):
    if element == value:
        raise ValueError("Expected value different from " + str(value) + " for " + element_name)