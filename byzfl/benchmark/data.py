"""Static dataset preparation for SNN benchmarks, before temporal encoding."""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_snn_transforms(base_transform, encoding_type):
    """Copy a vision transform chain, omitting normalization for spike coding."""
    if encoding_type not in ("constant", "rate", "latency"):
        raise ValueError(f"Unsupported SNN encoding: {encoding_type!r}")
    transform_list = list(base_transform.transforms)
    if encoding_type != "constant":
        transform_list = [t for t in transform_list if not isinstance(t, transforms.Normalize)]
    return transforms.Compose(transform_list)


def load_snn_data(params_manager, dataset_info):
    """Prepare train/validation/test data using an existing vision dataset entry.

    dataset_info is (torchvision class name, train transform, test transform).
    Returned samples remain static so data can be distributed before encoding.
    """
    model_config = params_manager.resolve_model_config()
    if not model_config["is_snn"]:
        raise ValueError("load_snn_data requires an SNN model.")
    encoding_type = model_config["encoding"]["type"]
    dataset_name, train_transform, test_transform = dataset_info
    dataset_class = getattr(datasets, dataset_name)
    train_base = dataset_class(
        root=params_manager.get_data_folder(), train=True, download=True,
        transform=get_snn_transforms(train_transform, encoding_type),
    )
    # Separate dataset instances prevent validation transforms from replacing
    # training augmentation through the shared dataset of two Subsets.
    val_base = dataset_class(
        root=params_manager.get_data_folder(), train=True, download=True,
        transform=get_snn_transforms(test_transform, encoding_type),
    )
    train_base.targets = torch.as_tensor(train_base.targets, dtype=torch.long)
    val_base.targets = torch.as_tensor(val_base.targets, dtype=torch.long)
    train_size = int(params_manager.get_size_train_set() * len(train_base))
    train_dataset, val_partition = random_split(
        train_base, [train_size, len(train_base) - train_size]
    )
    val_dataset = Subset(val_base, val_partition.indices)
    batch_size = params_manager.get_batch_size_evaluation()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if len(val_dataset) else None
    test_dataset = dataset_class(
        root=params_manager.get_data_folder(), train=False, download=True,
        transform=get_snn_transforms(test_transform, encoding_type),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_loader, test_loader
