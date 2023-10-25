import torch
import torch.linalg as linalg

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

def shuffle(vectors):
	vectors.data = vectors[torch.randperm(len(vectors))]

def reshape(vectors, dims):
	return torch.reshape(vectors, dims)
	
def concatenate(couple, axis = 0):
	return torch.concatenate(couple, axis)