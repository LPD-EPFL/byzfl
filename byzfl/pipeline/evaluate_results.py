import math
import json
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns

def custom_dict_to_str(d):
        if not d:
            return ''
        else:
            return str(d)

def find_best_hyperparameters(path_to_results, path_hyperparameters):
    try:
        with open(path_to_results+'/settings.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    training_seed = data["general"]["training_seed"]
    nb_training_seeds = data["general"]["nb_training_seeds"]
    nb_workers = data["general"]["nb_workers"]
    nb_byz = data["general"]["nb_byz"]
    nb_steps = data["general"]["nb_steps"] #Fixed
    evaluation_delta = data["general"]["evaluation_delta"] #Fixed

    data_distribution_seed = data["model"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["model"]["nb_data_distribution_seeds"]

    model_name = data["model"]["name"] #Fixed
    dataset_name = data["model"]["dataset_name"] #Fixed
    data_distributions = data["model"]["data_distribution"]
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    lr_list = data["honest_nodes"]["learning_rate"]
    momentum_list = data["honest_nodes"]["momentum"]
    wd_list = data["honest_nodes"]["weight_decay"]

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
    if len(pre_aggregators) == 0 or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    if isinstance(attacks, dict):
        attacks = [attacks]
    if isinstance(lr_list, float):
        lr_list = [lr_list]
    if isinstance(momentum_list, float):
        momentum_list = [momentum_list]
    if isinstance(wd_list, float):
        wd_list = [wd_list]

    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

    for nb_nodes in nb_workers:
        for nb_byzantine in nb_byz:
            if change:
                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
            for data_dist in data_distributions:
                distribution_parameter_list = data_dist["distribution_parameter"]
                for distribution_parameter in distribution_parameter_list:
                    for pre_agg in pre_aggregators:
                        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                        pre_agg_names = "_".join(pre_agg_list_names)
                        real_hyper_parameters = np.zeros((len(aggregators),3))
                        real_steps = np.zeros((len(aggregators), len(attacks)))
                        for k, agg in enumerate(aggregators):
                            max_acc_config = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list), len(attacks)))
                            hyper_parameters = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list), 3))
                            steps_max_reached = np.zeros((len(lr_list)*len(momentum_list)*len(wd_list), len(attacks)))
                            z = 0
                            for lr in lr_list:
                                for momentum in momentum_list:
                                    for wd in wd_list:
                                        tab_acc = np.zeros((
                                            len(attacks),
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        ))
                                        for i, attack in enumerate(attacks):
                                            for run_dd in range(nb_data_distribution_seeds):
                                                for run in range(nb_training_seeds):
                                                    common_name = str(dataset_name + "_" 
                                                                    + model_name + "_n_" 
                                                                    + str(nb_nodes) + "_f_" 
                                                                    + str(nb_byzantine) + "_d_"
                                                                    + str(nb_byzantine) + "_" 
                                                                    + custom_dict_to_str(data_dist["name"])  + "_"
                                                                    + str(distribution_parameter) + "_" 
                                                                    + custom_dict_to_str(agg["name"]) + "_" 
                                                                    + pre_agg_names + "_" 
                                                                    + custom_dict_to_str(attack["name"]) + "_lr_"
                                                                    + str(lr) + "_mom_"
                                                                    + str(momentum) + "_wd_"
                                                                    + str(wd))
                                                    tab_acc[i][run_dd][run] = genfromtxt(path_to_results+"/"+common_name+"/validation_accuracy_tr_seed_"+str(run+training_seed)+"_dd_seed_" + str(run_dd+data_distribution_seed) + ".txt", delimiter=',')

                                        for i, attack in enumerate(attacks):
                                            avg_accuracy = np.mean(tab_acc[i], axis=1)
                                            avg_accuracy_dd = np.mean(avg_accuracy, axis=0)
                                            idx_max = np.argmax(avg_accuracy_dd)
                                            max_acc_config[z][i] = avg_accuracy_dd[idx_max]
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
                                                            + str(nb_byzantine) + "_d_"
                                                            + str(nb_byzantine) + "_"
                                                            + custom_dict_to_str(data_dist["name"]) + "_"
                                                            + str(distribution_parameter) + "_"
                                                            + pre_agg_names + "_"
                                                            + agg["name"]
                                                            + ".txt")
                            np.savetxt(hyper_parameters_folder+"/"+file_name_hyperparameters, real_hyper_parameters[i])
                            
                            for j, attack in enumerate(attacks):
                                file_name_steps = str(dataset_name + "_"
                                                    + model_name + "_n_"
                                                    + str(nb_nodes) + "_f_"
                                                    + str(nb_byzantine) + "_d_"
                                                    + str(nb_byzantine) + "_"
                                                    + custom_dict_to_str(data_dist["name"]) + "_"
                                                    + str(distribution_parameter) + "_"
                                                    + pre_agg_names + "_"
                                                    + agg["name"] + "_"
                                                    + custom_dict_to_str(attack["name"])
                                                    + ".txt")
                                np.savetxt(steps_folder+"/"+file_name_steps, np.array([real_steps[i][j]]))

def heat_map_test_accuracy(path_to_results, path_to_hyperparameters, path_to_plot):
    try:
        with open(path_to_results+'/settings.json', 'r') as file:
            data = json.load(file)
    except Exception as e:
        print("ERROR: "+ str(e))

    if not os.path.exists(path_to_plot):
        os.makedirs(path_to_plot)
    
    training_seed = data["general"]["training_seed"]
    nb_training_seeds = data["general"]["nb_training_seeds"]
    nb_workers = data["general"]["nb_workers"]
    nb_byz = data["general"]["nb_byz"]
    nb_steps = data["general"]["nb_steps"] #Fixed
    evaluation_delta = data["general"]["evaluation_delta"] #Fixed

    data_distribution_seed = data["model"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["model"]["nb_data_distribution_seeds"]

    model_name = data["model"]["name"] #Fixed
    dataset_name = data["model"]["dataset_name"] #Fixed
    data_distributions = data["model"]["data_distribution"]
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

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
    if len(pre_aggregators) == 0 or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]
    if isinstance(attacks, dict):
        attacks = [attacks]

    for pre_agg in pre_aggregators:
        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)
        for agg in aggregators:
            for data_dist in data_distributions:
                distribution_parameter_list = data_dist["distribution_parameter"]
                heat_map_table = np.zeros((len(distribution_parameter_list), len(nb_byz)))
                for nb_nodes in nb_workers:
                    for y, nb_byzantine in enumerate(nb_byz):
                        if change:
                            nb_nodes = data["general"]["nb_honest"] + nb_byzantine
                        for x, dist_param in enumerate(distribution_parameter_list):
                            hyperparameters_file_name = str(dataset_name + "_"
                                                            + model_name + "_n_"
                                                            + str(nb_nodes) + "_f_"
                                                            + str(nb_byzantine) + "_d_"
                                                            + str(nb_byzantine) + "_"
                                                            + custom_dict_to_str(data_dist["name"]) + "_"
                                                            + str(dist_param) + "_"
                                                            + pre_agg_names + "_"
                                                            + agg["name"]
                                                            + ".txt")
                            
                            hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyperparameters_file_name)

                            lr = hyperparameters[0]
                            momentum = hyperparameters[1]
                            wd = hyperparameters[2]
                            
                            worst_accuracy = np.inf
                            for attack in attacks:
                                test_accuracy_file_name = str(
                                            dataset_name + "_" 
                                            + model_name + "_n_" 
                                            + str(nb_nodes) + "_f_" 
                                            + str(nb_byzantine) + "_d_"
                                            + str(nb_byzantine) + "_"
                                            + custom_dict_to_str(data_dist["name"]) + "_"
                                            + str(dist_param) + "_" 
                                            + custom_dict_to_str(agg["name"]) + "_"
                                            + pre_agg_names + "_"
                                            + custom_dict_to_str(attack["name"]) + "_" 
                                            + "lr_" + str(lr) + "_" 
                                            + "mom_" + str(momentum) + "_" 
                                            + "wd_" + str(wd)
                                )
                                try:
                                    with open(path_to_results+ "/" + test_accuracy_file_name +'/settings.json', 'r') as file:
                                        data = json.load(file)
                                except Exception as e:
                                    print("ERROR: "+ str(e))
                                nb_steps = data["general"]["nb_steps"]
                                nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
                                tab_acc = np.zeros((
                                        nb_data_distribution_seeds,
                                        nb_training_seeds, 
                                        nb_accuracies
                                ))
                                for run_dd in range(nb_data_distribution_seeds):
                                    for run in range(nb_training_seeds):
                                        tab_acc[run_dd][run] = genfromtxt(path_to_results + "/" + test_accuracy_file_name + "/test_accuracy_tr_seed_"+str(run+training_seed)+ "_dd_seed_" + str(run_dd + data_distribution_seed) + ".txt", delimiter=',')
                                tab_acc = tab_acc.mean(axis=1)
                                tab_acc = tab_acc.mean(axis=0)
                                accuracy = np.max(tab_acc)
                                if accuracy < worst_accuracy:
                                    worst_accuracy = accuracy
                                
                            heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                
                file_name = str(
                    "test_"
                    + dataset_name + "_"
                    + model_name + "_"
                    + custom_dict_to_str(data_dist["name"]) + "_"
                    + pre_agg_names + "_"
                    + agg["name"] + ".pdf"
                )
                
                column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()

                sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                plt.savefig(path_to_plot +"/"+ file_name)
                plt.close()

def best_heat_map_test_accuracy(path_to_results, path_to_hyperparameters, path_to_plot):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
        training_seed = data["general"]["training_seed"]
        nb_training_seeds = data["general"]["nb_training_seeds"]
        nb_workers = data["general"]["nb_workers"]
        nb_byz = data["general"]["nb_byz"]
        nb_steps = data["general"]["nb_steps"] #Fixed
        evaluation_delta = data["general"]["evaluation_delta"] #Fixed

        data_distribution_seed = data["model"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["model"]["nb_data_distribution_seeds"]

        model_name = data["model"]["name"] #Fixed
        dataset_name = data["model"]["dataset_name"] #Fixed
        data_distributions = data["model"]["data_distribution"]
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

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
        if len(pre_aggregators) == 0 or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]
        if isinstance(attacks, dict):
            attacks = [attacks]
        
        for pre_agg in pre_aggregators:
            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
            pre_agg_names = "_".join(pre_agg_list_names)
            heat_map_cube = np.zeros((len(aggregators), len(data_distributions[0]["distribution_parameter"]), len(nb_byz)))
            for z, agg in enumerate(aggregators):
                for data_dist in data_distributions:
                    distribution_parameter_list = data_dist["distribution_parameter"]
                    heat_map_table = np.zeros((len(distribution_parameter_list), len(nb_byz)))
                    for nb_nodes in nb_workers:
                        for y, nb_byzantine in enumerate(nb_byz):
                            if change:
                                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
                            for x, dist_param in enumerate(distribution_parameter_list):
                                hyperparameters_file_name = str(dataset_name + "_"
                                                                + model_name + "_n_"
                                                                + str(nb_nodes) + "_f_"
                                                                + str(nb_byzantine) + "_d_"
                                                                + str(nb_byzantine) + "_"
                                                                + custom_dict_to_str(data_dist["name"]) + "_"
                                                                + str(dist_param) + "_"
                                                                + pre_agg_names + "_"
                                                                + agg["name"]
                                                                + ".txt")
                                
                                hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyperparameters_file_name)
                                lr = hyperparameters[0]
                                momentum = hyperparameters[1]
                                wd = hyperparameters[2]
                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    test_accuracy_file_name = str(
                                                dataset_name + "_" 
                                                + model_name + "_n_" 
                                                + str(nb_nodes) + "_f_" 
                                                + str(nb_byzantine) + "_d_"
                                                + str(nb_byzantine) + "_"
                                                + custom_dict_to_str(data_dist["name"]) + "_"
                                                + str(dist_param) + "_" 
                                                + custom_dict_to_str(agg["name"]) + "_"
                                                + pre_agg_names + "_"
                                                + custom_dict_to_str(attack["name"]) + "_" 
                                                + "lr_" + str(lr) + "_" 
                                                + "mom_" + str(momentum) + "_" 
                                                + "wd_" + str(wd)
                                    )
                                    try:
                                        with open(path_to_results+ "/" + test_accuracy_file_name +'/settings.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))
                                    nb_steps = data["general"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
                                    tab_acc = np.zeros((
                                            nb_data_distribution_seeds,
                                            nb_training_seeds, 
                                            nb_accuracies
                                    ))
                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(path_to_results + "/" + test_accuracy_file_name + "/test_accuracy_tr_seed_"+str(run+training_seed)+ "_dd_seed_" + str(run_dd + data_distribution_seed) + ".txt", delimiter=',')
                                    tab_acc = tab_acc.mean(axis=1)
                                    tab_acc = tab_acc.mean(axis=0)
                                    accuracy = np.max(tab_acc)
                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

                    heat_map_cube[z] = heat_map_table
                    
                file_name = str(
                    "best_test_"
                    + dataset_name + "_"
                    + model_name + "_"
                    + custom_dict_to_str(data_dist["name"]) + "_"
                    + pre_agg_names + ".pdf"
                )
                
                column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()
                heat_map_table = np.max(heat_map_cube, axis=0)
                sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                plt.savefig(path_to_plot +"/"+ file_name)
                plt.close()

find_best_hyperparameters("./results", "./best_hyperparameters")
heat_map_test_accuracy("./results", "./best_hyperparameters", "./heat_maps")
best_heat_map_test_accuracy("./results", "./best_hyperparameters", "./heat_maps")