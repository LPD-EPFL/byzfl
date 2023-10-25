import numpy as np
import torch
import robust_aggregators_numpy as agg


tab_list_np = [np.array([1,2,3, 4]), np.array([3.5,2.5,7.5, 8.2]), np.array([4,5,6,7])]
tab_np = np.array([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])
tab_list_torch = [torch.FloatTensor([1,2,3, 4]), torch.FloatTensor([3.5,2.5,7.5, 8.2]), torch.FloatTensor([4,5,6,7])]
tab_torch = torch.FloatTensor([[1,2,3, 4], [3.5,2.5,7.5, 8.2], [4,5,6,7]])

print(agg.average(tab_list_np))
print(agg.average(tab_np))
print(agg.average(tab_list_torch))
print(agg.average(tab_torch))

