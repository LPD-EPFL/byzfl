import torch
import torch.linalg as linalg

def mean(vectors, axis=0):
	return torch.mean(vectors, axis=axis)

def median(vectors, axis = 0):
	return torch.quantile(vectors, dim=axis, q=0.5)
	# return torch.median(vectors, axis=axis)[0]

def sort(vectors, axis = 0):
	return torch.sort(vectors, axis=axis)[0]

def zeros_like(vector):
	return torch.zeros_like(vector)

def any(bools, axis=0):
	return torch.any(bools, axis=axis)

def isinf(vectors):
	return torch.isinf(vectors)

def sum(vectors, axis=0):
	return torch.sum(vectors, axis=axis)

def array(vectors):
	return torch.stack(vectors)

def argmin(vectors, axis=0):
	return torch.argmin(vectors, axis=axis)

def argmax(vectors, axis=0):
	return torch.argmax(vectors, axis=axis)

def argpartition(vectors, k, axis=0):
	return torch.topk(vectors, k+1, largest=False, dim=axis)[1]

def permutation(vectors):
	return vectors[torch.randperm(len(vectors))]

def shuffle(vectors):
	vectors.data = vectors[torch.randperm(len(vectors))]

def reshape(vectors, dims):
	return torch.reshape(vectors, dims)

def concatenate(couple, axis=0):
	return torch.concatenate(couple, axis)

def minimum(tensor1, tensor2):
	return torch.minimum(tensor1, tensor2)

def ones_like(tensor):
	return torch.ones_like(tensor)

def multiply(tensor1, tensor2):
	return torch.mul(tensor1, tensor2)

def divide(tensor1, tensor2):
	return torch.div(tensor1, tensor2)

def max(vectors):
	return torch.max(vectors)

def asarray(l):
	dtype = type(l[0])
	return torch.as_tensor(l, dtype=dtype)

def cdist(vector1, vector2):
	return torch.cdist(vector1, vector2)

def abs(vector):
	return torch.abs(vector)

def add(vector1, vector2):
	return torch.add(vector1, vector2)

def arange(d):
	return torch.arange(d)

def copy(vector):
	return vector.clone()

def stack(vectors):
	return torch.stack(vectors)

def var(vector, axis=0, ddof=1):
	return torch.var(vector, dim=axis, correction=ddof)

def sqrt(vector):
	return torch.sqrt(vector)

def full_like(vector, value, dtype):
	return torch.full_like(vector, value, dtype=torch.float64)

def dot(vector1, vector2):
	vector1 = vector1.double()
	vector2 = vector2.double()
	return torch.matmul(vector1, vector2)

def rand(vector):
	return torch.rand(vector.size(), dtype=vector.dtype, layout=vector.layout, device=vector.device)

def subtract(vector1, vector2):
	return torch.subtract(vector1, vector2)

#Sample size elements from a Gaussian distribution of mean loc and standard deviation scale. 
def normal(loc=0, scale=1, size=1):
	return torch.randn(size) * scale + loc