import torch

def mean(vectors: list or torch.Tensor, axis: None or int or tuple = None):
	if isinstance(vectors, list):
		vectors = torch.stack(vectors)
	return torch.mean(vectors, dim=axis)

def median(vectors: list or np.ndarray, axis: int = None):
	return np.median(vectors, axis=axis)