import math
import json
import os

import numpy as np
from numpy import genfromtxt
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

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

def custom_dict_to_str(d):
        if not d:
            return ''
        else:
            return str(d)

def evaluate_model(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return correct/total

def find_best_hyperparameters(path_to_results, path_hyperparameters):
    try:
        with open(path_to_results+'/settings.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    seed = data["general"]["seed"]
    nb_seeds = data["general"]["nb_seeds"]
    nb_workers = data["general"]["nb_workers"]
    nb_byz = data["general"]["nb_byz"]
    nb_steps = data["general"]["nb_steps"] #Fixed
    evaluation_delta = data["general"]["evaluation_delta"] #Fixed

    model_name = data["model"]["name"] #Fixed
    dataset_name = data["model"]["dataset_name"] #Fixed
    data_distributions = data["model"]["data_distribution"]
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    lr_list = data["honest_nodes"]["learning_rate"]
    momentum_list = data["honest_nodes"]["momentum"]
    wd_list = data["honest_nodes"]["weight_decay"]
    lr_decay = data["server"]["learning_rate_decay"]

    bit_precision = data["general"]["bit_precision"]

    attacks = data["attack"]
    
    if isinstance(nb_workers, int):
        nb_workers = [nb_workers]
    if not isinstance(nb_workers, list):
        nb_workers = [nb_workers]
        change = True
    if isinstance(nb_byz, int):
        nb_byz = [nb_byz]
    if isinstance(data_distributions, dict):
        data_distributions = [data_distributions]
    if isinstance(aggregators, dict):
        aggregators = [aggregators]
    if isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    if isinstance(attacks, dict):
        attacks = [attacks]
    if isinstance(lr_list, float):
        lr_list = [lr_list]
    if isinstance(momentum_list, float):
        momentum_list = [momentum_list]
    if isinstance(wd_list, float):
        wd_list = [wd_list]
    if isinstance(lr_decay, float):
        lr_decay = [lr_decay]
    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
    for nb_nodes in nb_workers:
        for nb_byzantine in nb_byz:
            if change:
                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
            for data_dist in data_distributions:
                alpha_list = [""]
                if data_dist["name"] == "dirichlet_niid":
                    alpha_list = data_dist["parameters"]["alpha"]
                if data_dist["name"] == "gamma_similarity_niid":
                    alpha_list = data_dist["parameters"]["gamma"]
                for alpha in alpha_list:
                    for pre_agg in pre_aggregators:
                        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                        pre_agg_names = "_".join(pre_agg_list_names)
                        real_hyper_parameters = np.zeros((len(aggregators),3))
                        real_steps = np.zeros((len(aggregators), len(attacks)))
                        for k, agg in enumerate(aggregators):
                            max_acc_config = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list)*len(lr_decay), len(attacks)))
                            hyper_parameters = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list)*len(lr_decay), 3))
                            steps_max_reached = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list)*len(lr_decay), len(attacks)))
                            z = 0
                            for lr in lr_list:
                                for momentum in momentum_list:
                                    for wd in wd_list:
                                        for lr_d in lr_decay:
                                            tab_acc = np.zeros((
                                                len(attacks), 
                                                nb_seeds, 
                                                nb_accuracies
                                            ))
                                            for i, attack in enumerate(attacks):
                                                for run in range(nb_seeds):
                                                    common_name = str(dataset_name + "_" 
                                                                    + model_name + "_n_" 
                                                                    + str(nb_nodes) + "_f_" 
                                                                    + str(nb_byzantine) + "_" 
                                                                    + custom_dict_to_str(data_dist["name"]) 
                                                                    + str(alpha) + "_" 
                                                                    + custom_dict_to_str(agg["name"]) + "_" 
                                                                    + pre_agg_names + "_" 
                                                                    + custom_dict_to_str(attack["name"]) + "_lr_"
                                                                    + str(lr) + "_mom_"
                                                                    + str(momentum) + "_wd_"
                                                                    + str(wd)+ "_lr_decay_"
                                                                    + str(lr_d))
                                                    tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')

                                            for i, attack in enumerate(attacks):
                                                avg_accuracy = np.mean(tab_acc[i], axis=0)
                                                idx_max = np.argmax(avg_accuracy)
                                                max_acc_config[z][i] = avg_accuracy[idx_max]
                                                steps_max_reached[z][i] = idx_max * evaluation_delta

                                            hyper_parameters[z][0] = lr
                                            hyper_parameters[z][1] = momentum
                                            hyper_parameters[z][2] = wd
                                            z += 1

                            if not os.path.exists(path_hyperparameters):
                                try:
                                    os.makedirs(path_hyperparameters)
                                except OSError as error:
                                    print(f"Error creating directory: {error}")

                            max_minimum_index = -1
                            maximum_minimum_value = -1
                            for i in range(z):
                                actual_min = np.min(max_acc_config[i])
                                if actual_min > maximum_minimum_value:
                                    max_minimum_index = i
                                    maximum_minimum_value = actual_min

                            real_hyper_parameters[k] = hyper_parameters[max_minimum_index]
                            real_steps[k] = steps_max_reached[max_minimum_index]

                        hyper_parameters_folder = path_hyperparameters + "/hyperparameters"
                        steps_folder = path_hyperparameters + "/better_step"

                        if not os.path.exists(hyper_parameters_folder):
                            os.makedirs(hyper_parameters_folder)

                        if not os.path.exists(steps_folder):
                            os.makedirs(steps_folder)

                        for i, agg in enumerate(aggregators):
                            file_name_hyperparameters = str(dataset_name + "_"
                                                            + model_name + "_n_"
                                                            + str(nb_nodes) + "_f_"
                                                            + str(nb_byzantine) + "_"
                                                            + custom_dict_to_str(data_dist["name"])
                                                            + str(alpha) + "_"
                                                            + pre_agg_names + "_"
                                                            + agg["name"]
                                                            + ".txt")
                            np.savetxt(hyper_parameters_folder+"/"+file_name_hyperparameters, real_hyper_parameters[i])
                            
                            for j, attack in enumerate(attacks):
                                file_name_steps = str(dataset_name + "_"
                                                    + model_name + "_n_"
                                                    + str(nb_nodes) + "_f_"
                                                    + str(nb_byzantine) + "_"
                                                    + custom_dict_to_str(data_dist["name"])
                                                    + str(alpha) + "_"
                                                    + pre_agg_names + "_"
                                                    + agg["name"] + "_"
                                                    + custom_dict_to_str(attack["name"])
                                                    + ".txt")
                                np.savetxt(steps_folder+"/"+file_name_steps, np.array([real_steps[i][j]]))

def evaluate_best_model_in_test(path_to_results, path_to_test_results, path_hyperparameters):
    try:
        with open(path_to_results+'/settings.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    if not os.path.exists(path_to_test_results):
        os.makedirs(path_to_test_results)
    
    with open(path_to_results+"settings.json", 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(',', ': '))
    
    seed = data["general"]["seed"]
    nb_seeds = data["general"]["nb_seeds"]
    nb_workers = data["general"]["nb_workers"]
    nb_byz = data["general"]["nb_byz"]
    device = data["general"]["device"]

    model_name = data["model"]["name"] #Fixed
    dataset_name = data["model"]["dataset_name"] #Fixed
    data_distributions = data["model"]["data_distribution"]
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    lr_decay = data["server"]["learning_rate_decay"]

    attacks = data["attack"]
    if isinstance(nb_workers, int):
        nb_workers = [nb_workers]
    if not isinstance(nb_workers, list):
        nb_workers = [nb_workers]
        change = True
    if isinstance(nb_byz, int):
        nb_byz = [nb_byz]
    if isinstance(data_distributions, dict):
        data_distributions = [data_distributions]
    if isinstance(aggregators, dict):
        aggregators = [aggregators]
    if isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    if isinstance(attacks, dict):
        attacks = [attacks]
    if isinstance(lr_decay, float):
        lr_decay = [lr_decay]
    
    data_folder = data["general"]["data_folder"]
    if data_folder is None:
        data_folder = "./data"
    dataset = getattr(torchvision.datasets, dict_[dataset_name][0])(
                root=data_folder, 
                train=False, 
                download=True,
                transform=dict_[dataset_name][2]
    )
    test_dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    for nb_nodes in nb_workers:
        for nb_byzantine in nb_byz:
            if change:
                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
            for lr_d in lr_decay:
                for data_dist in data_distributions:
                    alpha_list = [""]
                    if data_dist["name"] == "dirichlet_niid":
                        alpha_list = data_dist["parameters"]["alpha"]
                    if data_dist["name"] == "gamma_similarity_niid":
                        alpha_list = data_dist["parameters"]["alpha"]
                    for alpha in alpha_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:
                                hyperparameters_file_name = str(dataset_name + "_"
                                                                + model_name + "_n_"
                                                                + str(nb_nodes) + "_f_"
                                                                + str(nb_byzantine) + "_"
                                                                + custom_dict_to_str(data_dist["name"])
                                                                + str(alpha) + "_"
                                                                + pre_agg_names + "_"
                                                                + agg["name"]
                                                                + ".txt")
                                
                                hyperparameters = np.loadtxt(path_hyperparameters +"/"+ hyperparameters_file_name)
                                lr = hyperparameters[0]
                                momentum = hyperparameters[1]
                                if momentum == 0.0:
                                    momentum = 0
                                wd = hyperparameters[2]

                                for attack in attacks:
                                    tab_acc = np.zeros((
                                        nb_seeds
                                    ))
                                    for run in range(nb_seeds):
                                        common_name = str(dataset_name + "_" 
                                                        + model_name + "_n_" 
                                                        + str(nb_nodes) + "_f_" 
                                                        + str(nb_byzantine) + "_" 
                                                        + custom_dict_to_str(data_dist["name"]) 
                                                        + str(alpha) + "_" 
                                                        + custom_dict_to_str(agg["name"]) + "_" 
                                                        + pre_agg_names + "_" 
                                                        + custom_dict_to_str(attack["name"]) + "_lr_"
                                                        + str(lr) + "_mom_"
                                                        + str(momentum) + "_wd_"
                                                        + str(wd)+ "_lr_decay_"
                                                        + str(lr_d))
                                        
                                        model = torch.load(path_to_results+"/"+common_name+"models_seed_"+str(seed+run)+"/last_model.pt")
                                        tab_acc[run] = evaluate_model(model, test_dataloader, device)
                                    
                                    test_accuracy_file_name = str(
                                            path_to_results + "/"
                                            + dataset_name + "_" 
                                            + model_name + "_" 
                                            "n_" + str(nb_nodes) + "_" 
                                            + "f_" + str(nb_byzantine) + "_" 
                                            + custom_dict_to_str(data_dist["name"]) + "_"
                                            + str(alpha) + "_" 
                                            + custom_dict_to_str(agg["name"]) + "_"
                                            + pre_agg_names + "_"
                                            + custom_dict_to_str(attack["name"]) + "_" 
                                            + "lr_" + str(lr) + "_" 
                                            + "mom_" + str(momentum) + "_" 
                                            + "wd_" + str(wd) + "_" 
                                            + "lr_decay_" + str(lr_d) + "/"
                                        )
                                    np.savetxt(path_to_test_results+"/"+test_accuracy_file_name, [np.mean(tab_acc)], fmt='%.4f', delimiter=",")

find_best_hyperparameters("./results", "./best_hyperparameters")
#evaluate_best_model_in_test("./results/")