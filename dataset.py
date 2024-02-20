import torch, torchvision, random
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader

transforms_hflip = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
transforms_mnist = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
transforms_cifar_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_cifar_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dict_ = {
    "mnist":        ["MNIST", transforms_mnist, transforms_mnist],
    "fashionmnist": ["FashionMNIST", transforms_hflip, transforms_hflip],
    "emnist":       ["EMNIST", transforms_mnist, transforms_mnist],
    "cifar10":      ["CIFAR10", transforms_cifar_train, transforms_cifar_test],
    "cifar100":     ["CIFAR100", transforms_cifar_train, transforms_cifar_test],
    "imagenet":     ["ImageNet", transforms_hflip, transforms_hflip]
}

def generate_dataloaders(dataset_name,
                         data_folder = "./data",
                         nb_workers = 1,
                         nb_byz = 0,
                         data_dist = "iid",
                         alpha = None,
                         gamma = None,
                         training_batch_size = 64,
                         test_batch_size = 100):

    training_dataloaders = get_training_dataloaders(dataset_name,
                                                    data_folder,
                                                    nb_workers,
                                                    nb_byz,
                                                    data_dist,
                                                    alpha,
                                                    gamma,
                                                    training_batch_size)
    
    test_dataloader = get_test_dataloader(dataset_name,
                                          data_folder,
                                          test_batch_size)

    return training_dataloaders, test_dataloader

def get_training_dataloaders(dataset_name,
                             data_folder = "./data",
                             nb_workers = 1,
                             nb_byz = 0,
                             data_dist = "iid",
                             alpha = None,
                             gamma = None,
                             batch_size = 64):
    
    dataset = getattr(torchvision.datasets, dict_[dataset_name][0])(
            root = data_folder, 
            train = True, 
            download = True,
            transform = dict_[dataset_name][1]
    )

    nb_honest = nb_workers - nb_byz

    training_dataloaders = split_datasets(dataset,
                                          nb_honest, 
                                          data_dist, 
                                          alpha, 
                                          gamma,
                                          batch_size)
    return training_dataloaders

def get_test_dataloader(dataset_name, data_folder = "./data", batch_size = 100):    
    dataset = getattr(torchvision.datasets, dict_[dataset_name][0])(
                root=data_folder, 
                train=False, 
                download=True,
                transform=dict_[dataset_name][2]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def split_datasets(dataset, 
                   nb_honest = 1, 
                   data_dist = "iid", 
                   alpha = None, 
                   gamma = None,
                   batch_size = 64):

    targets = dataset.targets
    idx = list(range(len(targets)))

    match data_dist:
        case 'iid':
            split_idx = iid_idx(idx, nb_honest)
        case 'gamma_similarity_niid':
            split_idx = gamma_niid_idx(targets, idx, nb_honest, gamma)
        case 'dirichlet_niid':
            split_idx = dirichlet_niid_idx(targets, idx, nb_honest, alpha)
        case 'extreme_niid':
            split_idx = extreme_niid_idx(targets, idx, nb_honest)
        case _:
            raise ValueError(f"Invalid value for data_dist: {data_dist}")

    return idx_to_dataloaders(dataset, split_idx, batch_size)

def iid_idx(idx, nb_honest):
    random.shuffle(idx)
    split_idx = np.array_split(idx, nb_honest)
    return split_idx

def extreme_niid_idx(targets, idx, nb_honest):
    sorted_idx = np.array(sorted(zip(targets[idx],idx)))[:,1]
    split_idx = np.array_split(sorted_idx, nb_honest)
    return split_idx

def gamma_niid_idx(targets, idx, nb_honest, gamma):
    nb_similarity = int(len(idx)*gamma)
    iid = iid_idx(idx[:nb_similarity], nb_honest)
    niid = extreme_niid_idx(targets, idx[nb_similarity:], nb_honest)
    split_idx = [np.concatenate((iid[i],niid[i])) for i in range(nb_honest)]
    return split_idx

def dirichlet_niid_idx(targets, idx, nb_honest, alpha):
    c = len(torch.unique(targets))
    sample = np.random.dirichlet(np.repeat(alpha, nb_honest), size=c)
    p = np.cumsum(sample, axis=1)[:,:-1]
    idx = [np.where(targets == k)[0] for k in range(c)]
    idx = [np.split(idx[k], (p[k]*len(idx[k])).astype(int)) for k in range(c)]
    idx = [np.concatenate([idx[i][j] for i in range(c)]) for j in range(nb_honest)]
    return idx

def idx_to_dataloaders(dataset, split_idx, batch_size):
    data_loaders = []
    for i in range(len(split_idx)):
        subset = torch.utils.data.Subset(dataset, split_idx[i])
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        data_loaders.append(data_loader)
    return data_loaders