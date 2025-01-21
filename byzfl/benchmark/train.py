from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
from byzfl.benchmark.managers import ParamsManager

transforms_hflip = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transforms_cifar_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Supported datasets
dict_datasets = {
    "mnist":        ("MNIST", transforms_mnist, transforms_mnist),
    "fashionmnist": ("FashionMNIST", transforms_hflip, transforms_hflip),
    "emnist":       ("EMNIST", transforms_mnist, transforms_mnist),
    "cifar10":      ("CIFAR10", transforms_cifar_train, transforms_cifar_test),
    "cifar100":     ("CIFAR100", transforms_cifar_train, transforms_cifar_test),
    "imagenet":     ("ImageNet", transforms_hflip, transforms_hflip)
}


def start_training(params):
    params_manager = ParamsManager(params)

    # Configurations
    nb_honest_clients = params_manager.get_nb_honest()
    nb_byz_clients = params_manager.get_nb_byz()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_nodes_batch_size()

    set_random_seed(params_manager.get_data_distribution_seed())

    # Data Preparation
    key_dataset_name = params_manager.get_dataset_name()
    dataset_name = dict_datasets[key_dataset_name][0]
    dataset = getattr(datasets, dataset_name)(
            root = params_manager.get_data_folder(), 
            train = True, 
            download = True,
            transform = None
    )
    dataset.targets = Tensor(dataset.targets).long()

    train_size = int(params_manager.get_size_train_set() * len(dataset))
    val_size = len(dataset) - train_size

    # Split Train set into Train and Validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transformations to each dataset
    train_dataset.dataset.transform = dict_datasets[key_dataset_name][1]
    val_dataset.dataset.transform = dict_datasets[key_dataset_name][2]

    # Prepare Validation and Test data
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params_manager.get_server_batch_size_evaluation(), 
            shuffle=False
        )
    else:
        print("WARNING: NO VALIDATION DATASET")
    
    test_dataset = getattr(datasets, dataset_name)(
                root = params_manager.get_data_folder(),
                train=False, 
                download=True,
                transform=dict_datasets[key_dataset_name][2]
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=params_manager.get_server_batch_size_evaluation(), 
        shuffle=False
    )

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor({
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": params_manager.get_parameter_data_distribution(),
        "nb_honest": nb_honest_clients,
        "data_loader": train_dataset,
        "batch_size": batch_size,
    })
    client_dataloaders = data_distributor.split_data()

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": params_manager.get_model_name(),
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_server_optimizer_name(),
            "learning_rate": params_manager.get_server_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "weight_decay": params_manager.get_server_weight_decay(),
            "milestones": params_manager.get_server_milestones(),
            "learning_rate_decay": params_manager.get_server_learning_rate_decay(),
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[i],
            "momentum": params_manager.get_honest_nodes_momentum(),
            "nb_labels": params_manager.get_nb_labels(),
        }) for i in range(nb_honest_clients)
    ]

    # Server Setup, Use SGD Optimizer
    server = Server({
        "model_name": params_manager.get_model_name(),
        "device": params_manager.get_device(),
        "validation_loader": val_loader,
        "test_loader": test_loader,
        "optimizer_name": params_manager.get_server_optimizer_name(),
        "learning_rate": params_manager.get_server_learning_rate(),
        "weight_decay": params_manager.get_server_weight_decay(),
        "milestones": params_manager.get_server_milestones(),
        "learning_rate_decay": params_manager.get_server_learning_rate_decay(),
        "aggregator_info": params_manager.get_aggregator_info(),
        "pre_agg_list": params_manager.get_preaggregators(),
    })

    # Byzantine Client Setup
    attack = {
        "name": params_manager.get_attack_name(),
        "f": nb_byz_clients,
        "parameters": params_manager.get_attack_parameters(),
    }
    byz_client = ByzantineClient(attack)

    set_random_seed(params_manager.get_training_seed())

    # Training Loop
    for training_step in range(nb_training_steps):

        # Evaluate Global Model Every 100 Training Steps
        if training_step % 100 == 0:
            test_acc = server.compute_test_accuracy()
            print(f"--- Training Step {training_step}/{nb_training_steps} ---")
            print(f"Test Accuracy: {test_acc:.4f}")

        # Honest Clients Compute Gradients
        for client in honest_clients:
            client.compute_gradients()

        # Aggregate Honest Gradients
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

        # Apply Byzantine Attack
        byz_vector = byz_client.apply_attack(honest_gradients)

        # Combine Honest and Byzantine Gradients
        gradients = honest_gradients + byz_vector

        # Update Global Model
        server.update_model(gradients)

        # Send Updated Model to Clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)
    
    test_acc = server.compute_test_accuracy()
    print(f"--- Training Step {nb_training_steps}/{nb_training_steps} ---")
    print(f"Test Accuracy: {test_acc:.4f}")