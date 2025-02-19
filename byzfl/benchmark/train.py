import time
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
from byzfl.benchmark.managers import ParamsManager, FileManager

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

    # <----------------- File Manager  ----------------->
    file_manager = FileManager({
        "result_path": params_manager.get_results_directory(),
        "dataset_name": params_manager.get_dataset_name(),
        "model_name": params_manager.get_model_name(),
        "nb_workers": params_manager.get_nb_workers(),
        "nb_byz": params_manager.get_f(),
        "declared_nb_byz": params_manager.get_tolerated_f(),
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": params_manager.get_parameter_data_distribution(),
        "aggregation_name": params_manager.get_aggregator_name(),
        "pre_aggregation_names": [
            dict['name'] 
            for dict in params_manager.get_preaggregators()
        ],
        "attack_name": params_manager.get_attack_name(),
        "learning_rate": params_manager.get_server_learning_rate(),
        "momentum": params_manager.get_honest_nodes_momentum(),
        "weight_decay": params_manager.get_server_weight_decay(),
    })

    file_manager.save_config_dict(params_manager.get_data())

    # <----------------- Federated Framework ----------------->

    # Configurations
    nb_honest_clients = params_manager.get_nb_honest_clients()
    nb_byz_clients = params_manager.get_f()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_nodes_batch_size()

    dd_seed = params_manager.get_data_distribution_seed()
    training_seed = params_manager.get_training_seed()
    set_random_seed(dd_seed)

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

    attack_parameters = params_manager.get_attack_parameters()
    attack_parameters["aggregator_info"] = params_manager.get_aggregator_info()
    attack_parameters["pre_agg_list"] = params_manager.get_preaggregators()
    attack_parameters["f"] = nb_byz_clients

    label_flipping_attack = False
    attack_name = params_manager.get_attack_name()

    label_flipping_attack = attack_name == "LabelFlipping"

    attack = {
        "name": attack_name,
        "f": nb_byz_clients,
        "parameters": attack_parameters,
    }
    byz_client = ByzantineClient(attack)

    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    store_models = params_manager.get_store_models()
    store_training_loss = params_manager.get_store_training_loss()
    store_training_accuracy = params_manager.get_store_training_accuracy()

    val_accuracy_list = np.array([])
    test_accuracy_list = np.array([])

    start_time = time.time()

    # Send Initial Model to All Clients
    new_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(new_model)

    # Training Loop
    for training_step in range(nb_training_steps):

        # Evaluate Global Model Every Evaluation Delta Steps
        if training_step % evaluation_delta == 0:

            val_acc = server.compute_validation_accuracy()

            val_accuracy_list = np.append(val_accuracy_list, val_acc)

            file_manager.write_array_in_file(
                val_accuracy_list, 
                "val_accuracy_tr_seed_" + str(training_seed) 
                + "_dd_seed_" + str(dd_seed) +".txt"
            )

            if evaluate_on_test:
                test_acc = server.compute_test_accuracy()
                test_accuracy_list = np.append(test_accuracy_list, test_acc)

                file_manager.write_array_in_file(
                    test_accuracy_list, 
                    "test_accuracy_tr_seed_" + str(training_seed) 
                    + "_dd_seed_" + str(dd_seed) +".txt"
                )

            if store_models:
                file_manager.save_state_dict(
                    server.get_dict_parameters(),
                    training_seed,
                    dd_seed,
                    training_step
                )

        # Honest Clients Compute Gradients
        for client in honest_clients:
            client.compute_gradients()

        # Aggregate Honest Gradients
        honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

        # Deal with Label Flipping Attack
        attack_input = (
            [client.get_flat_flipped_gradients() for client in honest_clients]
            if label_flipping_attack
            else honest_gradients
        )

        # Apply Byzantine Attack
        byz_vector = byz_client.apply_attack(attack_input)

        # Combine Honest and Byzantine Gradients
        gradients = honest_gradients + byz_vector

        # Update Global Model
        server.update_model(gradients)

        # Send Updated Model to Clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)
    
    val_acc = server.compute_validation_accuracy()

    val_accuracy_list = np.append(val_accuracy_list, val_acc)

    file_manager.write_array_in_file(
        val_accuracy_list, 
        "val_accuracy_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )

    if evaluate_on_test:
        test_acc = server.compute_test_accuracy()
        test_accuracy_list = np.append(test_accuracy_list, test_acc)

        file_manager.write_array_in_file(
            test_accuracy_list, 
            "test_accuracy_tr_seed_" + str(training_seed) 
            + "_dd_seed_" + str(dd_seed) +".txt"
        )
    
    end_time = time.time()

    if store_training_loss:
        loss_list = [client.get_loss_list() for client in honest_clients]
        for client_id, loss in enumerate(loss_list):
            file_manager.save_loss(
                loss, 
                training_seed, 
                dd_seed, 
                client_id
            )
    
    if store_training_accuracy:
        train_acc_list = [client.get_train_accuracy() for client in honest_clients]
        for client_id, acc in enumerate(train_acc_list):
            file_manager.save_accuracy(
                acc, 
                training_seed, 
                dd_seed,
                client_id
            )
    
    if store_models:
        file_manager.save_state_dict(
            server.get_dict_parameters(),
            training_seed,
            dd_seed,
            training_step
        )
    
    execution_time = end_time - start_time

    file_manager.write_array_in_file(
        np.array(execution_time),
        "train_time_tr_seed_" + str(training_seed) 
        + "_dd_seed_" + str(dd_seed) +".txt"
    )