import numpy as np
import torch
import agg

torch.set_default_dtype(torch.float64)

tab_list_np = [np.array([1,2,3, 4]), np.array([3.5,2.5,7.5, 8.2]), np.array([4,5,6,7])]
tab_np = np.array([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])
tab_list_torch = [torch.DoubleTensor([1,2,3, 4]), torch.DoubleTensor([3.5,2.5,7.5, 8.2]), torch.DoubleTensor([4,5,6,7])]
tab_torch = torch.DoubleTensor([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])

print('-----------------------------------------------------------------------')
print('Average')
print('-----------------------------------------------------------------------')
test1 = agg.average(tab_list_np) 
test2 = agg.average(tab_np)
test3 = agg.average(tab_list_torch)
test4 = agg.average(tab_torch)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())

print('-----------------------------------------------------------------------')
print('Median')
print('-----------------------------------------------------------------------')
test1 = agg.median(tab_list_np) 
test2 = agg.median(tab_np)
test3 = agg.median(tab_list_torch)
test4 = agg.median(tab_torch)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Trmean')
print('-----------------------------------------------------------------------')
test1 = agg.trmean(tab_list_np, 1) 
test2 = agg.trmean(tab_np, 1)
test3 = agg.trmean(tab_list_torch, 1)
test4 = agg.trmean(tab_torch, 1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Geometric median')
print('-----------------------------------------------------------------------')
test1 = agg.geometric_median(tab_list_np) 
test2 = agg.geometric_median(tab_np)
test3 = agg.geometric_median(tab_list_torch)
test4 = agg.geometric_median(tab_torch)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())

print('-----------------------------------------------------------------------')
print('Krum')
print('-----------------------------------------------------------------------')
test1 = agg.krum(tab_list_np,1) 
test2 = agg.krum(tab_np,1)
test3 = agg.krum(tab_list_torch,1)
test4 = agg.krum(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Multi-Krum')
print('-----------------------------------------------------------------------')
test1 = agg.multi_krum(tab_list_np,1) 
test2 = agg.multi_krum(tab_np,1)
test3 = agg.multi_krum(tab_list_torch,1)
test4 = agg.multi_krum(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('NNM')
print('-----------------------------------------------------------------------')
test1 = agg.nnm(tab_list_np,1) 
test2 = agg.nnm(tab_np,1)
test3 = agg.nnm(tab_list_torch,1)
test4 = agg.nnm(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Bucketing')
print('-----------------------------------------------------------------------')
np.random.seed(1)
torch.manual_seed(1)
test1 = agg.bucketing(tab_list_np,2) 
np.random.seed(1)
torch.manual_seed(1)
test2 = agg.bucketing(tab_np,2)
np.random.seed(1)
torch.manual_seed(1)
test3 = agg.bucketing(tab_list_torch,2)
np.random.seed(1)
torch.manual_seed(1)
test4 = agg.bucketing(tab_torch,2)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


print('-----------------------------------------------------------------------')
print('Center Clipping')
print('-----------------------------------------------------------------------')
test1 = agg.centered_clipping(tab_list_np,np.mean(tab_list_np, axis = 0))
test2 = agg.centered_clipping(tab_np,np.mean(tab_np, axis = 0))
test3 = agg.centered_clipping(tab_list_torch,torch.mean(torch.stack(tab_list_torch), dim = 0))
test4 = agg.centered_clipping(tab_torch,torch.mean(tab_torch, dim = 0))

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())

print('-----------------------------------------------------------------------')
print('Minimum Diameter Averaging')
print('-----------------------------------------------------------------------')

test1 = agg.minimum_diameter_averaging(tab_list_np,1) 
test2 = agg.minimum_diameter_averaging(tab_np,1)
test3 = agg.minimum_diameter_averaging(tab_list_torch,1)
test4 = agg.minimum_diameter_averaging(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())

print('-----------------------------------------------------------------------')
print('Minimum Variance Averaging')
print('-----------------------------------------------------------------------')

test1 = agg.minimum_variance_averaging(tab_list_np,1) 
test2 = agg.minimum_variance_averaging(tab_np,1)
test3 = agg.minimum_variance_averaging(tab_list_torch,1)
test4 = agg.minimum_variance_averaging(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())

print('-----------------------------------------------------------------------')
print('Monna')
print('-----------------------------------------------------------------------')

test1 = agg.monna(tab_list_np,1) 
test2 = agg.monna(tab_np,1)
test3 = agg.monna(tab_list_torch,1)
test4 = agg.monna(tab_torch,1)

print(type(test1), type(test2), type(test3), type(test4))
print(test1)
print(test2)
print(test3.numpy())
print(test4.numpy())


