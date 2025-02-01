import math
import json
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns


def custom_dict_to_str(dictionary):
    """
    Safely convert a dictionary to a string.
    Returns an empty string if the dictionary is empty.
    """
    return '' if not dictionary else str(dictionary)


def ensure_list(value):
    """
    Ensure the given value is returned as a list.
    If it is not, wrap it in a list.
    """
    if not isinstance(value, list):
        value = [value]
    return value


def compute_number_of_honest_workers(nb_workers, nb_byz, fix_workers_as_honest):
    """
    Compute the number of honest and total workers based on:
      - Initial nb_workers
      - Number of byzantine workers (nb_byz)
      - Whether the worker set is fixed to be honest (fix_workers_as_honest)

    Returns:
      (total_workers, honest_workers)
    """
    if fix_workers_as_honest:
        nb_honest = nb_workers
        nb_workers = nb_honest + nb_byz
    else:
        nb_honest = nb_workers - nb_byz

    return nb_workers, nb_honest


def find_best_hyperparameters(path_to_results, path_hyperparameters):
    """
    Find the best hyperparameters (learning rate, momentum, weight decay) 
    that maximize the minimum accuracy across different attacks.

    Reads a configuration file (config.json) in `path_to_results` 
    and writes out the best hyperparameters and the corresponding 
    step at which maximum accuracy was reached for each aggregator 
    and each attack.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_workers = data["benchmark_config"]["nb_workers"]
    nb_byz = data["benchmark_config"]["nb_byz"]
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    fix_workers_as_honest = data["benchmark_config"]["fix_workers_as_honest"]

    # <-------------- Server Config ------------->
    nb_steps = data["server"]["nb_steps"]
    lr_list = data["server"]["learning_rate"]
    wd_list = data["server"]["weight_decay"]

    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_nodes"]["momentum"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_workers = ensure_list(nb_workers)
    nb_byz = ensure_list(nb_byz)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    # Number of accuracy checkpoints
    nb_accuracies = 1 + math.ceil(nb_steps / evaluation_delta)

    # Main nested loops to explore configurations
    for nb_nodes in nb_workers:
        for nb_byzantine in nb_byz:
            # If fix_workers_as_honest is True, total nodes = nb_nodes + nb_byzantine
            # otherwise, it's just nb_nodes with nb_byzantine byzantine among them
            if fix_workers_as_honest:
                nb_nodes += nb_byzantine

            for data_dist in data_distributions:
                distribution_parameter_list = ensure_list(data_dist["distribution_parameter"])
                for distribution_parameter in distribution_parameter_list:
                    for pre_agg in pre_aggregators:
                        # Build a single name from all pre-aggregators
                        pre_agg_names_list = [p["name"] for p in pre_agg]
                        pre_agg_names = "_".join(pre_agg_names_list)

                        # Prepare arrays to store final best hyperparams & steps
                        real_hyper_parameters = np.zeros((len(aggregators), 3))
                        real_steps = np.zeros((len(aggregators), len(attacks)))

                        for k, agg in enumerate(aggregators):
                            # We'll store max accuracy for each (lr, momentum, wd) across attacks
                            num_combinations = len(lr_list) * len(momentum_list) * len(wd_list)
                            max_acc_config = np.zeros((num_combinations, len(attacks)))
                            hyper_parameters = np.zeros((num_combinations, 3))
                            steps_max_reached = np.zeros((num_combinations, len(attacks)))

                            index_combination = 0
                            for lr in lr_list:
                                for momentum in momentum_list:
                                    for wd in wd_list:
                                        # tab_acc shape: (len(attacks), nb_dd_seeds, nb_training_seeds, nb_accuracies)
                                        tab_acc = np.zeros(
                                            (
                                                len(attacks),
                                                nb_data_distribution_seeds,
                                                nb_training_seeds,
                                                nb_accuracies
                                            )
                                        )

                                        # Fill tab_acc with loaded accuracy files
                                        for i, attack in enumerate(attacks):
                                            for run_dd in range(nb_data_distribution_seeds):
                                                for run in range(nb_training_seeds):
                                                    file_name = (
                                                        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                        f"d_{nb_byzantine}_{custom_dict_to_str(data_dist['name'])}_"
                                                        f"{distribution_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                        f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                        f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                                    )
                                                    acc_path = os.path.join(
                                                        path_to_results,
                                                        file_name,
                                                        f"val_accuracy_tr_seed_{run + training_seed}"
                                                        f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                                    )
                                                    tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                        tab_acc = tab_acc.reshape(
                                            len(attacks),
                                            nb_data_distribution_seeds * nb_training_seeds,
                                            nb_accuracies
                                        )
                                        
                                        # Compute average accuracy across seeds, find max
                                        for i in range(len(attacks)):
                                            avg_accuracy = np.mean(tab_acc[i], axis=0)
                                            idx_max = np.argmax(avg_accuracy)
                                            max_acc_config[index_combination, i] = avg_accuracy[idx_max]
                                            steps_max_reached[index_combination, i] = idx_max * evaluation_delta

                                        hyper_parameters[index_combination] = [lr, momentum, wd]
                                        index_combination += 1

                            # Create path if needed
                            if not os.path.exists(path_hyperparameters):
                                try:
                                    os.makedirs(path_hyperparameters)
                                except OSError as error:
                                    print(f"Error creating directory: {error}")

                            # Find the combination that maximizes the minimum accuracy across attacks
                            max_minimum_idx = -1
                            max_minimum_val = -1
                            for i in range(num_combinations):
                                current_min = np.min(max_acc_config[i])
                                if current_min > max_minimum_val:
                                    max_minimum_idx = i
                                    max_minimum_val = current_min

                            real_hyper_parameters[k] = hyper_parameters[max_minimum_idx]
                            real_steps[k] = steps_max_reached[max_minimum_idx]

                        # Save results to folder
                        hyper_parameters_folder = os.path.join(path_hyperparameters, "hyperparameters")
                        steps_folder = os.path.join(path_hyperparameters, "better_step")

                        os.makedirs(hyper_parameters_folder, exist_ok=True)
                        os.makedirs(steps_folder, exist_ok=True)

                        for i, agg in enumerate(aggregators):
                            # Save best hyperparameters
                            file_name_hparams = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                f"d_{nb_byzantine}_{custom_dict_to_str(data_dist['name'])}_"
                                f"{distribution_parameter}_{pre_agg_names}_{agg['name']}.txt"
                            )
                            np.savetxt(
                                os.path.join(hyper_parameters_folder, file_name_hparams),
                                real_hyper_parameters[i]
                            )

                            # Save step at which max accuracy occurs for each attack
                            for j, attack in enumerate(attacks):
                                file_name_steps = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                    f"d_{nb_byzantine}_{custom_dict_to_str(data_dist['name'])}_"
                                    f"{distribution_parameter}_{pre_agg_names}_{agg['name']}_"
                                    f"{custom_dict_to_str(attack['name'])}.txt"
                                )
                                step_val = np.array([real_steps[i, j]])
                                np.savetxt(os.path.join(steps_folder, file_name_steps), step_val)

colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (120/255,120/255, 120/255), (0.7, 0.2, 0.5)]
tab_sign = ['-', '--', '-.', ':', 'solid']
markers = ['^','s', '<', 'o', '*']

def plot_accuracy_fix_agg_best_setting(path_to_results, path_to_hyperparameters, path_to_plot, colors=colors, tab_sign=tab_sign, markers=markers):
        
        try:
            with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"ERROR reading config.json: {e}")
            return
        
        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        

        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_workers = data["benchmark_config"]["nb_workers"]
        nb_byz = data["benchmark_config"]["nb_byz"]
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        fix_workers_as_honest = data["benchmark_config"]["fix_workers_as_honest"]

        # <-------------- Server Config ------------->
        nb_steps = data["server"]["nb_steps"]
        lr_list = data["server"]["learning_rate"]
        wd_list = data["server"]["weight_decay"]

        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]

        # <-------------- Honest Nodes Config ------------->
        momentum_list = data["honest_nodes"]["momentum"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        nb_workers = ensure_list(nb_workers)
        nb_byz = ensure_list(nb_byz)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        wd_list = ensure_list(wd_list)

        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

        for nb_nodes in nb_workers:
            for nb_byzantine in nb_byz:
                if fix_workers_as_honest:
                    nb_nodes += nb_byzantine
                for data_dist in data_distributions:
                    dist_parameter_list = data_dist["distribution_parameter"]
                    for dist_parameter in dist_parameter_list:
                        for pre_agg in pre_aggregators:
                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                            pre_agg_names = "_".join(pre_agg_list_names)
                            for agg in aggregators:

                                hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_{pre_agg_names}_{agg['name']}.txt"
                                )

                                hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyper_file_name)
                                lr = hyperparameters[0]
                                momentum = hyperparameters[1]
                                wd = hyperparameters[2]

                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_data_distribution_seeds,
                                    nb_training_seeds,
                                    nb_accuracies
                                ))

                                for i, attack in enumerate(attacks):
                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            file_name = (
                                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
                                                f"d_{nb_byzantine}_{custom_dict_to_str(data_dist['name'])}_"
                                                f"{dist_parameter}_{custom_dict_to_str(agg['name'])}_"
                                                f"{pre_agg_names}_{custom_dict_to_str(attack['name'])}_"
                                                f"lr_{lr}_mom_{momentum}_wd_{wd}"
                                            )
                                            acc_path = os.path.join(
                                                path_to_results,
                                                file_name,
                                                f"test_accuracy_tr_seed_{run + training_seed}"
                                                f"_dd_seed_{run_dd + data_distribution_seed}.txt"
                                            )
                                            tab_acc[i, run_dd, run] = genfromtxt(acc_path, delimiter=',')

                                tab_acc = tab_acc.reshape(
                                    len(attacks),
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_accuracies
                                )
                                
                                err = np.zeros((len(attacks), nb_accuracies))
                                for i in range(len(err)):
                                    err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_training_seeds*nb_data_distribution_seeds)
                                
                                plt.rcParams.update({'font.size': 12})

                                
                                for i, attack in enumerate(attacks):
                                    attack = attack["name"]
                                    plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = attack, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                    plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                plt.xlabel('Round')
                                plt.ylabel('Accuracy')
                                plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                plt.ylim(0,1)
                                plt.grid()
                                plt.legend()

                                plot_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_parameter}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )

                                plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                plt.close()


def heat_map_loss( path_to_results, path_to_hyperparameters, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best training losses reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_workers = data["benchmark_config"]["nb_workers"]
    nb_byz = data["benchmark_config"]["nb_byz"]
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    fix_workers_as_honest = data["benchmark_config"]["fix_workers_as_honest"]

    # <-------------- Server Config ------------->
    nb_steps = data["server"]["nb_steps"]
    lr_list = data["server"]["learning_rate"]
    wd_list = data["server"]["weight_decay"]

    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_nodes"]["momentum"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_workers = ensure_list(nb_workers)
    nb_byz = ensure_list(nb_byz)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                heat_map_table = np.zeros((len(distribution_parameter_list), len(nb_byz)))

                for nb_nodes in nb_workers:
                    for y, nb_byzantine in enumerate(nb_byz):

                        if fix_workers_as_honest:
                            nb_honest = nb_nodes
                            nb_nodes += nb_byzantine
                        else:
                            nb_honest = nb_nodes - nb_byzantine

                        for x, dist_param in enumerate(distribution_parameter_list):

                            hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                            )

                            
                            hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyper_file_name)

                            lr = hyperparameters[0]
                            momentum = hyperparameters[1]
                            wd = hyperparameters[2]
                            
                            lowest_loss = 0
                            for attack in attacks:

                                config_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                    f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )

                                try:
                                    with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                        data = json.load(file)
                                except Exception as e:
                                    print("ERROR: "+ str(e))

                                nb_steps = data["server"]["nb_steps"]

                                losses = np.zeros(
                                    (
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_honest,
                                        nb_steps
                                    )
                                )

                                for run_dd in range(nb_data_distribution_seeds):
                                    for run in range(nb_training_seeds):
                                        for worker_i in range(nb_honest):
                                            losses[run_dd][run][worker_i] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"loss_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}"
                                                f"/loss_client_{worker_i}.txt",
                                                delimiter=','
                                            )

                                losses = losses.reshape(
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_honest,
                                    nb_steps
                                )

                                losses = np.mean(losses, axis=1)
                                losses = np.mean(losses, axis=0)


                                temp_lowest_loss = np.min(losses)

                                if temp_lowest_loss > lowest_loss:
                                    lowest_loss = temp_lowest_loss
                                
                            heat_map_table[len(heat_map_table)-1-x][y] = lowest_loss
                
                file_name = (
                    f"loss_{dataset_name}_"
                    f"{model_name}_"
                    f"{custom_dict_to_str(data_dist['name'])}_"
                    f"{pre_agg_names}_"
                    f"{agg['name']}.pdf"
                )

                
                column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()

                try:
                    os.makedirs(path_to_plot, exist_ok=True)
                except OSError as error:
                    print(f"Error creating directory: {error}")

                sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, cmap=sns.cm.rocket_r, annot=True)
                plt.savefig(path_to_plot +"/"+ file_name)
                plt.close()


def heat_map_test_accuracy(path_to_results, path_to_hyperparameters, path_to_plot):
    """
    Creates a heatmap where the axis are the number of 
    byzantine nodes and the distribution parameter.
    Each number is the mean of the best accuracy reached 
    by the model across seeds, using a specific aggregation.
    """
    try:
        with open(os.path.join(path_to_results, 'config.json'), 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"ERROR reading config.json: {e}")
        return
    
    try:
        os.makedirs(path_to_plot, exist_ok=True)
    except OSError as error:
        print(f"Error creating directory: {error}")
    

    # <-------------- Benchmark Config ------------->
    training_seed = data["benchmark_config"]["training_seed"]
    nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
    nb_workers = data["benchmark_config"]["nb_workers"]
    nb_byz = data["benchmark_config"]["nb_byz"]
    data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
    nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
    data_distributions = data["benchmark_config"]["data_distribution"]
    fix_workers_as_honest = data["benchmark_config"]["fix_workers_as_honest"]

    # <-------------- Server Config ------------->
    nb_steps = data["server"]["nb_steps"]
    lr_list = data["server"]["learning_rate"]
    wd_list = data["server"]["weight_decay"]

    # <-------------- Evaluation and Results ------------->
    evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

    # <-------------- Model Config ------------->
    model_name = data["model"]["name"]
    dataset_name = data["model"]["dataset_name"]

    # <-------------- Honest Nodes Config ------------->
    momentum_list = data["honest_nodes"]["momentum"]

    # <-------------- Aggregators Config ------------->
    aggregators = data["aggregator"]
    pre_aggregators = data["pre_aggregators"]

    # <-------------- Attacks Config ------------->
    attacks = data["attack"]

    # Ensure certain configurations are always lists
    nb_workers = ensure_list(nb_workers)
    nb_byz = ensure_list(nb_byz)
    data_distributions = ensure_list(data_distributions)
    aggregators = ensure_list(aggregators)

    # Pre-aggregators can be multiple or single dict; unify them
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        pre_aggregators = [pre_aggregators]

    attacks = ensure_list(attacks)
    lr_list = ensure_list(lr_list)
    momentum_list = ensure_list(momentum_list)
    wd_list = ensure_list(wd_list)

    for pre_agg in pre_aggregators:

        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
        pre_agg_names = "_".join(pre_agg_list_names)

        for agg in aggregators:

            for data_dist in data_distributions:

                distribution_parameter_list = data_dist["distribution_parameter"]
                heat_map_table = np.zeros((len(distribution_parameter_list), len(nb_byz)))

                for nb_nodes in nb_workers:
                    for y, nb_byzantine in enumerate(nb_byz):

                        if fix_workers_as_honest:
                            nb_nodes += nb_byzantine

                        for x, dist_param in enumerate(distribution_parameter_list):

                            hyper_file_name = (
                                f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_{pre_agg_names}_{agg['name']}.txt"
                            )

                            
                            hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyper_file_name)

                            lr = hyperparameters[0]
                            momentum = hyperparameters[1]
                            wd = hyperparameters[2]
                            
                            worst_accuracy = np.inf
                            for attack in attacks:

                                config_file_name = (
                                    f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                    f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                    f"{custom_dict_to_str(attack['name'])}_lr_{lr}_mom_{momentum}_wd_{wd}"
                                )

                                try:
                                    with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                        data = json.load(file)
                                except Exception as e:
                                    print("ERROR: "+ str(e))

                                nb_steps = data["server"]["nb_steps"]
                                nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                tab_acc = np.zeros(
                                    (
                                        nb_data_distribution_seeds,
                                        nb_training_seeds,
                                        nb_accuracies
                                    )
                                )

                                for run_dd in range(nb_data_distribution_seeds):
                                    for run in range(nb_training_seeds):
                                        tab_acc[run_dd][run] = genfromtxt(
                                            f"{path_to_results}/{config_file_name}/"
                                            f"test_accuracy_tr_seed_{run + training_seed}_"
                                            f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                            delimiter=','
                                        )

                                tab_acc = tab_acc.reshape(
                                    nb_data_distribution_seeds * nb_training_seeds,
                                    nb_accuracies
                                )

                                tab_acc = tab_acc.mean(axis=0)

                                accuracy = np.max(tab_acc)

                                if accuracy < worst_accuracy:
                                    worst_accuracy = accuracy
                                
                            heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                
                file_name = (
                    f"test_{dataset_name}_"
                    f"{model_name}_"
                    f"{custom_dict_to_str(data_dist['name'])}_"
                    f"{pre_agg_names}_"
                    f"{agg['name']}.pdf"
                )

                
                column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()

                try:
                    os.makedirs(path_to_plot, exist_ok=True)
                except OSError as error:
                    print(f"Error creating directory: {error}")

                sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                plt.savefig(path_to_plot +"/"+ file_name)
                plt.close()


def aggregated_heat_map_test_accuracy(path_to_results, path_to_hyperparameters, path_to_plot):
        """
        Heatmap with the aggregated info of all aggregators, 
        for every region in the heatmap, it shows the aggregation 
        with the best accuracy.
        """
        try:
            with open(path_to_results+'/config.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        try:
            os.makedirs(path_to_plot, exist_ok=True)
        except OSError as error:
            print(f"Error creating directory: {error}")
        
        # <-------------- Benchmark Config ------------->
        training_seed = data["benchmark_config"]["training_seed"]
        nb_training_seeds = data["benchmark_config"]["nb_training_seeds"]
        nb_workers = data["benchmark_config"]["nb_workers"]
        nb_byz = data["benchmark_config"]["nb_byz"]
        data_distribution_seed = data["benchmark_config"]["data_distribution_seed"]
        nb_data_distribution_seeds = data["benchmark_config"]["nb_data_distribution_seeds"]
        data_distributions = data["benchmark_config"]["data_distribution"]
        fix_workers_as_honest = data["benchmark_config"]["fix_workers_as_honest"]

        # <-------------- Server Config ------------->
        nb_steps = data["server"]["nb_steps"]
        lr_list = data["server"]["learning_rate"]
        wd_list = data["server"]["weight_decay"]

        # <-------------- Evaluation and Results ------------->
        evaluation_delta = data["evaluation_and_results"]["evaluation_delta"]

        # <-------------- Model Config ------------->
        model_name = data["model"]["name"]
        dataset_name = data["model"]["dataset_name"]

        # <-------------- Honest Nodes Config ------------->
        momentum_list = data["honest_nodes"]["momentum"]

        # <-------------- Aggregators Config ------------->
        aggregators = data["aggregator"]
        pre_aggregators = data["pre_aggregators"]

        # <-------------- Attacks Config ------------->
        attacks = data["attack"]

        # Ensure certain configurations are always lists
        nb_workers = ensure_list(nb_workers)
        nb_byz = ensure_list(nb_byz)
        data_distributions = ensure_list(data_distributions)
        aggregators = ensure_list(aggregators)

        # Pre-aggregators can be multiple or single dict; unify them
        if not pre_aggregators or isinstance(pre_aggregators[0], dict):
            pre_aggregators = [pre_aggregators]

        attacks = ensure_list(attacks)
        lr_list = ensure_list(lr_list)
        momentum_list = ensure_list(momentum_list)
        wd_list = ensure_list(wd_list)
        
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

                            if fix_workers_as_honest:
                                nb_nodes += nb_byzantine

                            for x, dist_param in enumerate(distribution_parameter_list):

                                hyper_file_name = (
                                    f"{dataset_name}_"
                                    f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                    f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                    f"{pre_agg_names}_{agg['name']}.txt"
                                )

                                
                                hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyper_file_name)
                                lr = hyperparameters[0]
                                momentum = hyperparameters[1]
                                wd = hyperparameters[2]
                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    config_file_name = (
                                        f"{dataset_name}_"
                                        f"{model_name}_n_{nb_nodes}_f_{nb_byzantine}_d_{nb_byzantine}_"
                                        f"{custom_dict_to_str(data_dist['name'])}_{dist_param}_"
                                        f"{custom_dict_to_str(agg['name'])}_{pre_agg_names}_"
                                        f"{custom_dict_to_str(attack['name'])}_"
                                        f"lr_{lr}_"
                                        f"mom_{momentum}_"
                                        f"wd_{wd}"
                                    )

                                    try:
                                        with open(path_to_results+ "/" + config_file_name +'/config.json', 'r') as file:
                                            data = json.load(file)
                                    except Exception as e:
                                        print("ERROR: "+ str(e))

                                    nb_steps = data["server"]["nb_steps"]
                                    nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

                                    tab_acc = np.zeros(
                                        (
                                            nb_data_distribution_seeds,
                                            nb_training_seeds,
                                            nb_accuracies
                                        )
                                    )

                                    for run_dd in range(nb_data_distribution_seeds):
                                        for run in range(nb_training_seeds):
                                            tab_acc[run_dd][run] = genfromtxt(
                                                f"{path_to_results}/{config_file_name}/"
                                                f"test_accuracy_tr_seed_{run + training_seed}_"
                                                f"dd_seed_{run_dd + data_distribution_seed}.txt",
                                                delimiter=','
                                            )

                                    tab_acc = tab_acc.reshape(
                                        nb_data_distribution_seeds * nb_training_seeds,
                                        nb_accuracies
                                    )
                                    
                                    tab_acc = tab_acc.mean(axis=0)
                                    accuracy = np.max(tab_acc)

                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy

                    heat_map_cube[z] = heat_map_table
                    
                file_name = (
                    f"best_test_{dataset_name}_"
                    f"{model_name}_"
                    f"{custom_dict_to_str(data_dist['name'])}_"
                    f"{pre_agg_names}.pdf"
                )
                
                column_names = [str(dist_param) for dist_param in distribution_parameter_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()

                heat_map_table = np.max(heat_map_cube, axis=0)
                sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names, annot=True)
                plt.savefig(path_to_plot +"/"+ file_name)
                plt.close()