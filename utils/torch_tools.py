import torch
import torch.linalg as linalg
import numpy as np

def mean(vectors, axis = 0):
	return torch.mean(vectors, axis=axis)

def median(vectors, axis = 0):
	return torch.median(vectors, axis=axis)[0]

def sort(vectors, axis = 0):
	return torch.sort(vectors, axis=axis)[0]

def zeros_like(vector):
	return torch.zeros_like(vector)

def any(bools, axis = 0):
	return torch.any(bools, axis = axis)

def isinf(vectors):
	return torch.isinf(vectors)

def sum(vectors, axis = 0):
	return torch.sum(vectors, axis = axis)

def array(vectors):
	return torch.stack(vectors)

def argmin(vectors, axis = 0):
	return torch.argmin(vectors, axis = axis)

def argpartition(vectors, k, axis=0):
	return torch.topk(vectors, k, largest=False, dim = axis)[1]

def permutation(vectors):
	return vectors[torch.randperm(len(vectors))]

def shuffle(vectors):
	vectors.data = vectors[torch.randperm(len(vectors))]

def reshape(vectors, dims):
	return torch.reshape(vectors, dims)

def concatenate(couple, axis = 0):
	return torch.concatenate(couple, axis)

def minimum(tensor1, tensor2):
	return torch.minimum(tensor1, tensor2)

def ones_like(tensor):
	return torch.ones_like(tensor)

def multiply(tensor1, tensor2):
	return torch.multiply(tensor1, tensor2)

def max(vectors):
	return torch.max(vectors)

def asarray(l):
	print(l)
	dtype = type(l[0])
	return torch.as_tensor(l, dtype = dtype)




