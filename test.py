import numpy as np
import torch
import agg
import sys
import random

sys.path.append('../papers/ByzLibrary')
from robust_aggregators import RobustAggregator, average_nearest_neighbors

n = 10
d = 100000
tab = np.random.rand(n,d)
# tab[n//2,d//2] = np.inf

# print(tab)


# randomlist = random.sample(range(10, 30), 5)
# print(randomlist)
# sys.exit(-1)

tab_list_np = [np.copy(tab[i]) for i in range(len(tab))]
tab_np = np.array(np.copy(tab))
tab_list_torch = [torch.from_numpy(np.copy(tab[i])) for i in range(len(tab))]
tab_torch = torch.from_numpy(np.copy(tab))

# print(type(tab))
# print(type(tab_list_np))
# print(type(tab_np))
# print(type(tab_list_torch))
# print(type(tab_torch))

# sys.exit(-1)

# tab_list_np = [np.array([1,2,3, 4]), np.array([3.5,2.5,7.5, 8.2]), np.array([4,5,6,7])]
# tab_np = np.array([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])
# tab_list_torch = [torch.DoubleTensor([1,2,3, 4]), torch.DoubleTensor([3.5,2.5,7.5, 8.2]), torch.DoubleTensor([4,5,6,7])]
# tab_torch = torch.DoubleTensor([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])

print('-----------------------------------------------------------------------')
print('Average')
test1 = agg.average(tab_list_np) 
test2 = agg.average(tab_np)
test3 = agg.average(tab_list_torch)
test4 = agg.average(tab_torch)
test5 = RobustAggregator("average", None, n, 0, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))
# print(np.any(np.isclose(test2, test3.numpy(), rtol = 1e-7)))
# print(np.any(np.isclose(test3.numpy(), test4.numpy(), rtol = 1e-7)))
# print(np.any(np.isclose(test4.numpy(), test1, rtol = 1e-7)))
# print(test2 == test3.numpy())
# print(test3.numpy() == test4.numpy())

print('-----------------------------------------------------------------------')
print('Median')
test1 = agg.median(tab_list_np) 
test2 = agg.median(tab_np)
test3 = agg.median(tab_list_torch)
test4 = agg.median(tab_torch)
test5 = RobustAggregator("median", None, n, 0, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))

print('-----------------------------------------------------------------------')
print('Trmean')
test1 = agg.trmean(tab_list_np, 1) 
test2 = agg.trmean(tab_np, 1)
test3 = agg.trmean(tab_list_torch, 1)
test4 = agg.trmean(tab_torch, 1)
test5 = RobustAggregator("trmean", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())
print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))


print('-----------------------------------------------------------------------')
print('Geometric median')
# print('-----------------------------------------------------------------------')
test1 = agg.geometric_median(tab_list_np) 
test2 = agg.geometric_median(tab_np)
test3 = agg.geometric_median(tab_list_torch)
test4 = agg.geometric_median(tab_torch)
test5 = RobustAggregator("geometric_median_old", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))

print('-----------------------------------------------------------------------')
print('Krum')
# print('-----------------------------------------------------------------------')
test1 = agg.krum(tab_list_np,1) 
test2 = agg.krum(tab_np,1)
test3 = agg.krum(tab_list_torch,1)
test4 = agg.krum(tab_torch,1)
test5 = RobustAggregator("krum", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))



print('-----------------------------------------------------------------------')
print('Multi-Krum')
# print('-----------------------------------------------------------------------')

test1 = agg.multi_krum(tab_list_np,1)
test2 = agg.multi_krum(tab_np,1)
test3 = agg.multi_krum(tab_list_torch,1)
test4 = agg.multi_krum(tab_torch,1)
test5 = RobustAggregator("multi_krum", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))


print('-----------------------------------------------------------------------')
print('NNM')
# print('-----------------------------------------------------------------------')
test1 = agg.nnm(tab_list_np,1) 
test2 = agg.nnm(tab_np,1)
test3 = agg.nnm(tab_list_torch,1)
test4 = agg.nnm(tab_torch,1)
mixed_vectors = list()
for vector in tab_list_torch:
    mixed_vectors.append(average_nearest_neighbors(tab_list_torch, 1, vector))
mixed_vectors = torch.stack(mixed_vectors)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), mixed_vectors.numpy(), atol = 1e-7)) and np.any(np.isclose(mixed_vectors.numpy(), test1, atol = 1e-7)))

# print('-----------------------------------------------------------------------')
# print('Bucketing')
# print('-----------------------------------------------------------------------')
# np.random.seed(1)
# torch.manual_seed(1)
# test1 = agg.bucketing(tab_list_np,2) 
# np.random.seed(1)
# torch.manual_seed(1)
# test2 = agg.bucketing(tab_np,2)
# np.random.seed(1)
# torch.manual_seed(1)
# test3 = agg.bucketing(tab_list_torch,2)
# np.random.seed(1)
# torch.manual_seed(1)
# test4 = agg.bucketing(tab_torch,2)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Center Clipping')
# print('-----------------------------------------------------------------------')
test1 = agg.centered_clipping(tab_list_np,np.zeros(d))
test2 = agg.centered_clipping(tab_np,np.zeros(d))
test3 = agg.centered_clipping(tab_list_torch,torch.zeros(d))
test4 = agg.centered_clipping(tab_torch,torch.zeros(d))

test5 = RobustAggregator("cc", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())
# print(test5.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))

print('-----------------------------------------------------------------------')
print('Minimum Diameter Averaging')
# print('-----------------------------------------------------------------------')

test1 = agg.mda(tab_list_np,1) 
test2 = agg.mda(tab_np,1)
test3 = agg.mda(tab_list_torch,1)
test4 = agg.mda(tab_torch,1)
test5 = RobustAggregator("mda", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))


print('-----------------------------------------------------------------------')
print('Minimum Variance Averaging')
# print('-----------------------------------------------------------------------')

test1 = agg.mva(tab_list_np,1) 
test2 = agg.mva(tab_np,1)
test3 = agg.mva(tab_list_torch,1)
test4 = agg.mva(tab_torch,1)
test5 = RobustAggregator("mva", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)

# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())

print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))

print('-----------------------------------------------------------------------')
print('Monna')
# print('-----------------------------------------------------------------------')

test1 = agg.monna(tab_list_np,1) 
test2 = agg.monna(tab_np,1)
test3 = agg.monna(tab_list_torch,1)
test4 = agg.monna(tab_torch,1)
test5 = RobustAggregator("monna", None, n, 1, None, d, "cpu").aggregate(tab_list_torch)


# print(type(test1), type(test2), type(test3), type(test4))
# print(test1)
# print(test2)
# print(test3.numpy())
# print(test4.numpy())
print(np.any(np.isclose(test1, test2, atol = 1e-7)) and np.any(np.isclose(test2, test3.numpy(), atol = 1e-7)) and np.any(np.isclose(test3.numpy(), test4.numpy(), atol = 1e-7)) and np.any(np.isclose(test4.numpy(), test5.numpy(), atol = 1e-7)) and np.any(np.isclose(test5.numpy(), test1, atol = 1e-7)))


