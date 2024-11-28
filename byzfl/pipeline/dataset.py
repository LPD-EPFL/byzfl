import torch, torchvision, random
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader, random_split

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

def get_dataloaders(params):

    dataset_name = params["dataset_name"]
    dataset = getattr(torchvision.datasets, dict_[dataset_name][0])(
            root = params["data_folder"], 
            train = True, 
            download = True,
            transform = dict_[dataset_name][1]
    )
    
    dataset.targets = torch.Tensor(dataset.targets).long()
    train_size = int(params["size_train_set"] * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    validation_dataloader = None

    if len(val_dataset) > 0:

        validation_dataloader = DataLoader(val_dataset, 
                                        batch_size=params["batch_size_validation"], 
                                        shuffle=True)
        
    else:
        print("WARNING: NO VALIDATION DATASET")

    nb_honest = params["nb_workers"] - params["nb_byz"]
    
    params_split_datasets = {
        "nb_honest": nb_honest,
        "data_distribution_name": params["data_distribution_name"],
        "distribution_parameter": params["distribution_parameter"],
        "batch_size": params["batch_size"],
        "dataset": train_dataset
    }

    training_dataloaders = split_datasets(params_split_datasets)

    test_dataset = getattr(torchvision.datasets, dict_[dataset_name][0])(
                root = params["data_folder"],
                train=False, 
                download=True,
                transform=dict_[dataset_name][1]
    )

    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size_validation"], shuffle=True)

    return training_dataloaders, validation_dataloader, test_dataloader

def split_datasets(params):
    data_dist = params["data_distribution_name"]
    distribution_parameter = params["distribution_parameter"]
    nb_honest = params["nb_honest"] 
    dataset = params["dataset"]
    #targets = dataset.dataset.targets[dataset.indices]
    targets = dataset.dataset.targets
    idx = dataset.indices

    match data_dist:
        case 'iid':
            split_idx = iid_idx(idx, nb_honest)
        case 'gamma_similarity_niid':
            split_idx = gamma_niid_idx(targets, idx, nb_honest, distribution_parameter)
        case 'dirichlet_niid':
            split_idx = dirichlet_niid_idx(targets, idx, nb_honest, distribution_parameter)
        case 'extreme_niid':
            split_idx = extreme_niid_idx(targets, idx, nb_honest)
        case _:
            raise ValueError(f"Invalid value for data_dist: {data_dist}")
    
    return idx_to_dataloaders(dataset, split_idx, params["batch_size"])

def iid_idx(idx, nb_honest):
    random.shuffle(idx)
    split_idx = np.array_split(idx, nb_honest)
    return split_idx

def extreme_niid_idx(targets, idx, nb_honest):
    if len(idx) == 0:
        return list([[]]*nb_honest)
    sorted_idx = np.array(sorted(zip(targets[idx],idx)))[:,1]
    split_idx = np.array_split(sorted_idx, nb_honest)
    return split_idx

def gamma_niid_idx(targets, idx, nb_honest, gamma):
    nb_similarity = int(len(idx)*gamma)
    iid = iid_idx(idx[:nb_similarity], nb_honest)
    niid = extreme_niid_idx(targets, idx[nb_similarity:], nb_honest)
    split_idx = [np.concatenate((iid[i],niid[i])) for i in range(nb_honest)]
    split_idx = [node_idx.astype(int) for node_idx in split_idx]
    return split_idx

def dirichlet_niid_idx(targets, idx, nb_honest, alpha):
    c = len(torch.unique(targets))
    sample = np.random.dirichlet(np.repeat(alpha, nb_honest), size=c)
    p = np.cumsum(sample, axis=1)[:,:-1]
    aux_idx = [np.where(targets[idx] == k)[0] for k in range(c)]
    aux_idx = [np.split(aux_idx[k], (p[k]*len(aux_idx[k])).astype(int)) for k in range(c)]
    aux_idx = [np.concatenate([aux_idx[i][j] for i in range(c)]) for j in range(nb_honest)]
    idx = np.array(idx)
    aux_idx = [list(idx[aux_idx[i]]) for i in range(len(aux_idx))]
    return aux_idx

def idx_to_dataloaders(dataset, split_idx, batch_size):
    data_loaders = []
    for i in range(len(split_idx)):
        subset = torch.utils.data.Subset(dataset.dataset, split_idx[i])
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        data_loaders.append(data_loader)
    return data_loaders
