import math
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import seaborn as sns

class Plot(object):
    
    def custom_dict_to_str(self, d):
        if not d:
            return ''
        else:
            return str(d)
    
    def plot_accuracy_fix_agg(self, path_to_results, path_to_plot):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
        tab_sign = ['-', '--', '-.', ':', 'solid']
        markers = ['^','s', '<', 'o', '*']

        colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (120/255,120/255, 120/255), (0.7, 0.2, 0.5)]
        
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

        attacks = data["attack"]
        if isinstance(nb_workers, int):
            nb_workers = [nb_workers]
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
                for lr in lr_list:
                    for momentum in momentum_list:
                        for wd in wd_list:
                            for lr_d in lr_decay:
                                for data_dist in data_distributions:
                                    alpha = ""
                                    if data_dist["name"] == "dirichlet_niid":
                                        alpha = data_dist["parameters"]["alpha"]
                                    for agg in aggregators:
                                        for pre_agg in pre_aggregators:
                                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                            pre_agg_names = "_".join(pre_agg_list_names)
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
                                                                    + self.custom_dict_to_str(data_dist["name"]) 
                                                                    + str(alpha) + "_" 
                                                                    + self.custom_dict_to_str(agg["name"]) + "_" 
                                                                    + pre_agg_names + "_" 
                                                                    + self.custom_dict_to_str(attack["name"]) + "_lr_"
                                                                    + str(lr) + "_mom_"
                                                                    + str(momentum) + "_wd_"
                                                                    + str(wd)+ "_lr_decay_"
                                                                    + str(lr_d))
                                                    tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                            
                                            err = np.zeros((len(attacks), nb_accuracies))
                                            for i in range(len(err)):
                                                err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_seeds)
                                            
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

                                            plot_name = str(dataset_name + "_"
                                                            + model_name + "_n_"
                                                            + str(nb_nodes) + "_f_"
                                                            + str(nb_byzantine) + "_"
                                                            + self.custom_dict_to_str(data_dist["name"])
                                                            + str(alpha) + "_"
                                                            + self.custom_dict_to_str(agg["name"]) + "_"
                                                            + pre_agg_names + "_lr_"
                                                            + str(lr) + "_mom_"
                                                            + str(momentum) + "_wd_"
                                                            + str(wd)+ "_lr_decay_"
                                                            + str(lr_d))
                                            plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                            plt.close()
    
    def plot_accuracy_fix_attack(self, path_to_results, path_to_plot):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
        tab_sign = ['-', '--', '-.', ':', 'solid']
        markers = ['^','s', '<', 'o', '*']

        colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (120/255,120/255, 120/255), (0.7, 0.2, 0.5)]
        
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

        attacks = data["attack"]
        if isinstance(nb_workers, int):
            nb_workers = [nb_workers]
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
                for lr in lr_list:
                    for momentum in momentum_list:
                        for wd in wd_list:
                            for lr_d in lr_decay:
                                for data_dist in data_distributions:
                                    alpha = ""
                                    if data_dist["name"] == "dirichlet_niid":
                                        alpha = data_dist["parameters"]["alpha"]
                                    for attack in attacks:
                                        for pre_agg in pre_aggregators:
                                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                            pre_agg_names = "_".join(pre_agg_list_names)
                                            tab_acc = np.zeros((
                                                len(aggregators), 
                                                nb_seeds, 
                                                nb_accuracies
                                            ))
                                            for i, agg in enumerate(aggregators):
                                                for run in range(nb_seeds):
                                                    common_name = str(dataset_name + "_" 
                                                                    + model_name + "_n_" 
                                                                    + str(nb_nodes) + "_f_" 
                                                                    + str(nb_byzantine) + "_" 
                                                                    + self.custom_dict_to_str(data_dist["name"]) 
                                                                    + str(alpha) + "_" 
                                                                    + self.custom_dict_to_str(agg["name"]) + "_" 
                                                                    + pre_agg_names + "_" 
                                                                    + self.custom_dict_to_str(attack["name"]) + "_lr_"
                                                                    + str(lr) + "_mom_"
                                                                    + str(momentum) + "_wd_"
                                                                    + str(wd)+ "_lr_decay_"
                                                                    + str(lr_d))
                                                    tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                            
                                            err = np.zeros((len(aggregators), nb_accuracies))
                                            for i in range(len(err)):
                                                err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_seeds)
                                            
                                            plt.rcParams.update({'font.size': 12})

                                            
                                            for i, agg in enumerate(aggregators):
                                                agg = agg["name"]
                                                plt.plot(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0), label = agg, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1)
                                                plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, np.mean(tab_acc[i], axis = 0) - err[i], np.mean(tab_acc[i], axis = 0) + err[i], alpha = 0.25)

                                            plt.xlabel('Round')
                                            plt.ylabel('Accuracy')
                                            plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                            plt.ylim(0,1)
                                            plt.grid()
                                            plt.legend()

                                            plot_name = str(dataset_name + "_"
                                                            + model_name + "_n_"
                                                            + str(nb_nodes) + "_f_"
                                                            + str(nb_byzantine) + "_"
                                                            + self.custom_dict_to_str(data_dist["name"])
                                                            + str(alpha) + "_"
                                                            + str(attack["name"]) + "_"
                                                            + pre_agg_names + "_lr_"
                                                            + str(lr) + "_mom_"
                                                            + str(momentum) + "_wd_"
                                                            + str(wd)+ "_lr_decay_"
                                                            + str(lr_d))
                                            plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                            plt.close()
    
    def plot_minimum_of_max_accuracies(self, path_to_results, path_to_plot):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
        tab_sign = ['-', '--', '-.', ':', 'solid']
        markers = ['^','s', '<', 'o', '*']

        colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (0.5, 0.5, 0.5), (0.7, 0.2, 0.5)]
        
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

        attacks = data["attack"]
        if isinstance(nb_workers, int):
            nb_workers = [nb_workers]
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
                for lr in lr_list:
                    for momentum in momentum_list:
                        for wd in wd_list:
                            for lr_d in lr_decay:
                                for data_dist in data_distributions:
                                    alpha = ""
                                    if data_dist["name"] == "dirichlet_niid":
                                        alpha = data_dist["parameters"]["alpha"]
                                    for pre_agg in pre_aggregators:
                                        pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                        pre_agg_names = "_".join(pre_agg_list_names)
                                        minimum_max = np.array([np.inf]*len(aggregators))
                                        tab_acc_min_max = np.zeros((
                                                len(aggregators),
                                                nb_accuracies
                                        ))
                                        tab_err_min_max = np.zeros((
                                                len(aggregators),
                                                nb_accuracies
                                        ))
                                        for attack in attacks:
                                            tab_acc = np.zeros((
                                                len(aggregators), 
                                                nb_seeds, 
                                                nb_accuracies
                                            ))
                                            for i, agg in enumerate(aggregators):
                                                for run in range(nb_seeds):
                                                    common_name = str(dataset_name + "_" 
                                                                    + model_name + "_n_" 
                                                                    + str(nb_nodes) + "_f_" 
                                                                    + str(nb_byzantine) + "_" 
                                                                    + self.custom_dict_to_str(data_dist["name"]) 
                                                                    + str(alpha) + "_" 
                                                                    + self.custom_dict_to_str(agg["name"]) + "_" 
                                                                    + pre_agg_names + "_" 
                                                                    + self.custom_dict_to_str(attack["name"]) + "_lr_"
                                                                    + str(lr) + "_mom_"
                                                                    + str(momentum) + "_wd_"
                                                                    + str(wd)+ "_lr_decay_"
                                                                    + str(lr_d))
                                                    tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                            
                                            err = np.zeros((len(aggregators), nb_accuracies))
                                            for i in range(len(err)):
                                                err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_seeds)
                                            
                                            plt.rcParams.update({'font.size': 12})

                                            for i, agg in enumerate(aggregators):
                                                mean_accuraccy_agg = np.mean(tab_acc[i], axis = 0)
                                                max_value = np.max(mean_accuraccy_agg)
                                                if max_value < minimum_max[i]:
                                                    minimum_max[i] = max_value
                                                    tab_acc_min_max[i] = mean_accuraccy_agg
                                                    tab_err_min_max[i] = err[i]

                                        plt.rcParams.update({'font.size': 12})

                                            
                                        for i, agg in enumerate(aggregators):
                                            agg = agg["name"]
                                            plt.plot(np.arange(nb_accuracies)*evaluation_delta, tab_acc_min_max[i], label = agg, color = colors[i], linestyle = tab_sign[i], marker = markers[i], markevery = 1, alpha=0.8)
                                            plt.fill_between(np.arange(nb_accuracies)*evaluation_delta, tab_acc_min_max[i] - tab_err_min_max[i], tab_acc_min_max[i] + tab_err_min_max[i], alpha = 0.25)

                                        plt.xlabel('Round')
                                        plt.ylabel('Accuracy')
                                        plt.xlim(0,(nb_accuracies-1)*evaluation_delta)
                                        plt.ylim(0,1)
                                        plt.grid()
                                        plt.legend()

                                        plot_name = str(dataset_name + "_"
                                                        + model_name + "_n_"
                                                        + str(nb_nodes) + "_f_"
                                                        + str(nb_byzantine) + "_"
                                                        + self.custom_dict_to_str(data_dist["name"])
                                                        + str(alpha) + "_"
                                                        + pre_agg_names + "_lr_"
                                                        + str(lr) + "_mom_"
                                                        + str(momentum) + "_wd_"
                                                        + str(wd)+ "_lr_decay_"
                                                        + str(lr_d))
                                        plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                        plt.close()
    
    def plot_accuracy_fix_agg_best_setting(self, path_to_results, path_to_plot, path_to_hyperparameters):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
        tab_sign = ['-', '--', '-.', ':', 'solid']
        markers = ['^','s', '<', 'o', '*']

        colors = [(0, 0.4470, 0.7410), (0.8500, 0.3250, 0.0980), (0.4660, 0.6740, 0.1880), (120/255,120/255, 120/255), (0.7, 0.2, 0.5)]
        
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
        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
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
                            alpha_list = data_dist["parameters"]["gamma"]
                        for alpha in alpha_list:
                            for pre_agg in pre_aggregators:
                                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                pre_agg_names = "_".join(pre_agg_list_names)
                                for agg in aggregators:
                                    hyperparameters_file_name = str(dataset_name + "_"
                                                                    + model_name + "_n_"
                                                                    + str(nb_nodes) + "_f_"
                                                                    + str(nb_byzantine) + "_"
                                                                    + self.custom_dict_to_str(data_dist["name"])
                                                                    + str(alpha) + "_"
                                                                    + pre_agg_names + "_"
                                                                    + agg["name"]
                                                                    + ".txt")
                                    
                                    hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyperparameters_file_name)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    if momentum == 0.0:
                                        momentum = 0
                                    wd = hyperparameters[2]

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
                                                            + self.custom_dict_to_str(data_dist["name"]) 
                                                            + str(alpha) + "_" 
                                                            + self.custom_dict_to_str(agg["name"]) + "_" 
                                                            + pre_agg_names + "_" 
                                                            + self.custom_dict_to_str(attack["name"]) + "_lr_"
                                                            + str(lr) + "_mom_"
                                                            + str(momentum) + "_wd_"
                                                            + str(wd)+ "_lr_decay_"
                                                            + str(lr_d))
                                            tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                    
                                    err = np.zeros((len(attacks), nb_accuracies))
                                    for i in range(len(err)):
                                        err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_seeds)
                                    
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

                                    plot_name = str(dataset_name + "_"
                                                    + model_name + "_n_"
                                                    + str(nb_nodes) + "_f_"
                                                    + str(nb_byzantine) + "_"
                                                    + self.custom_dict_to_str(data_dist["name"])
                                                    + str(alpha) + "_"
                                                    + self.custom_dict_to_str(agg["name"]) + "_"
                                                    + pre_agg_names + "_lr_"
                                                    + str(lr) + "_mom_"
                                                    + str(momentum) + "_wd_"
                                                    + str(wd)+ "_lr_decay_"
                                                    + str(lr_d))
                                    plt.savefig(path_to_plot+"/"+plot_name+'_plot.pdf')
                                    plt.close()
    
    def heat_map_test_accuracy(self, path_to_test_results, path_to_hyperparameters, path_to_plot):
        try:
            with open(path_to_test_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
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
        lr_d = 1.0
        for pre_agg in pre_aggregators:
            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
            pre_agg_names = "_".join(pre_agg_list_names)
            for agg in aggregators:
                for data_dist in data_distributions:
                    alpha_list = [""]
                    if data_dist["name"] == "dirichlet_niid":
                        alpha_list = data_dist["parameters"]["alpha"]
                    if data_dist["name"] == "gamma_similarity_niid":
                        alpha_list = data_dist["parameters"]["alpha"]
                    #try:
                    heat_map_table = np.zeros((len(alpha_list), len(nb_byz)))
                    for nb_nodes in nb_workers:
                        for y, nb_byzantine in enumerate(nb_byz):
                            if change:
                                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
                                for x, alpha in enumerate(alpha_list):
                                    hyperparameters_file_name = str(dataset_name + "_"
                                                                    + model_name + "_n_"
                                                                    + str(nb_nodes) + "_f_"
                                                                    + str(nb_byzantine) + "_"
                                                                    + self.custom_dict_to_str(data_dist["name"])
                                                                    + str(alpha) + "_"
                                                                    + pre_agg_names + "_"
                                                                    + agg["name"]
                                                                    + ".txt")
                                    
                                    hyperparameters = np.loadtxt(path_to_hyperparameters +"/"+ hyperparameters_file_name)
                                    lr = hyperparameters[0]
                                    momentum = hyperparameters[1]
                                    if momentum == 0.0:
                                        momentum = 0
                                    wd = hyperparameters[2]
                                    
                                    worst_accuracy = np.inf

                                    for attack in attacks:
                                        
                                        test_accuracy_file_name = str(
                                                path_to_test_results + "/"
                                                + dataset_name + "_" 
                                                + model_name + "_" 
                                                "n_" + str(nb_nodes) + "_" 
                                                + "f_" + str(nb_byzantine) + "_" 
                                                + self.custom_dict_to_str(data_dist["name"]) + "_"
                                                + str(alpha) + "_" 
                                                + self.custom_dict_to_str(agg["name"]) + "_"
                                                + pre_agg_names + "_"
                                                + self.custom_dict_to_str(attack["name"]) + "_" 
                                                + "lr_" + str(lr) + "_" 
                                                + "mom_" + str(momentum) + "_" 
                                                + "wd_" + str(wd) + "_" 
                                                + "lr_decay_" + str(lr_d) + "/"
                                            )
                                        accuracy = np.load(path_to_test_results + "/" + test_accuracy_file_name)

                                        if accuracy < worst_accuracy:
                                            worst_accuracy = accuracy
                                    
                                    heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                    
                    file_name = str(
                        dataset_name + "_"
                        + model_name + "_"
                        + self.custom_dict_to_str(data_dist["name"])
                        + str(alpha) + "_"
                        + pre_agg_names + "_"
                        + agg["name"] + ".pdf"
                    )
                    
                    row_names = [str(alpha) for alpha in alpha_list]
                    column_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                    
                    sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names)
                    plt.savefig(path_to_plot +"/"+ file_name)
                    
                    #except Exception as e:
                    #    print(e)
    
    def heat_map_validation(self, path_to_results, path_to_plot, path_to_hyperparameters):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_plot):
            os.makedirs(path_to_plot)
        
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
        lr_d = 1.0
        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
        for pre_agg in pre_aggregators:
            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
            pre_agg_names = "_".join(pre_agg_list_names)
            for agg in aggregators:
                for data_dist in data_distributions:
                    alpha_list = [""]
                    if data_dist["name"] == "dirichlet_niid":
                        alpha_list = data_dist["parameters"]["alpha"]
                    if data_dist["name"] == "gamma_similarity_niid":
                        alpha_list = data_dist["parameters"]["gamma"]
                    #try:
                    heat_map_table = np.zeros((len(alpha_list), len(nb_byz)))
                    for nb_nodes in nb_workers:
                        for y, nb_byzantine in enumerate(nb_byz):
                            if change:
                                nb_nodes = data["general"]["nb_honest"] + nb_byzantine
                            for x, alpha in enumerate(alpha_list):
                                hyperparameters_file_name = str(dataset_name + "_"
                                                                + model_name + "_n_"
                                                                + str(nb_nodes) + "_f_"
                                                                + str(nb_byzantine) + "_"
                                                                + self.custom_dict_to_str(data_dist["name"])
                                                                + str(alpha) + "_"
                                                                + pre_agg_names + "_"
                                                                + agg["name"]
                                                                + ".txt")
                                
                                hyperparameters = np.loadtxt(path_to_hyperparameters +"/hyperparameters/"+ hyperparameters_file_name)
                                lr = hyperparameters[0]
                                momentum = hyperparameters[1]
                                if momentum == 0.0:
                                    momentum = 0
                                wd = hyperparameters[2]
                                
                                worst_accuracy = np.inf
                                for attack in attacks:
                                    tab_acc = np.zeros((
                                            nb_seeds, 
                                            nb_accuracies
                                    ))
                                    for run in range(nb_seeds):
                                        test_accuracy_file_name = str(
                                                dataset_name + "_" 
                                                + model_name + "_" 
                                                "n_" + str(nb_nodes) + "_" 
                                                + "f_" + str(nb_byzantine) + "_" 
                                                + self.custom_dict_to_str(data_dist["name"])
                                                + str(alpha) + "_" 
                                                + self.custom_dict_to_str(agg["name"]) + "_"
                                                + pre_agg_names + "_"
                                                + self.custom_dict_to_str(attack["name"]) + "_" 
                                                + "lr_" + str(lr) + "_" 
                                                + "mom_" + str(momentum) + "_" 
                                                + "wd_" + str(wd) + "_" 
                                                + "lr_decay_" + str(lr_d) + "/"
                                            )
                                        tab_acc[run] = genfromtxt(path_to_results + "/" + test_accuracy_file_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                    accuracy = np.max(tab_acc)
                                    if accuracy < worst_accuracy:
                                        worst_accuracy = accuracy
                                    
                                heat_map_table[len(heat_map_table)-1-x][y] = worst_accuracy
                    
                    file_name = str(
                        dataset_name + "_"
                        + model_name + "_"
                        + self.custom_dict_to_str(data_dist["name"]) + "_"
                        + pre_agg_names + "_"
                        + agg["name"] + ".pdf"
                    )
                    
                    column_names = [str(alpha) for alpha in alpha_list]
                    row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                    column_names.reverse()

                    sns.heatmap(heat_map_table, xticklabels=row_names, yticklabels=column_names)
                    plt.savefig(path_to_plot +"/"+ file_name)
                    plt.close()
                    #except Exception as e:
                    #    print(e)

plot = Plot()
"""
plot.plot_minimum_of_max_accuracies("./results", "./min_max_plots")
plot.plot_accuracy_fix_attack("./results", "./fix_attack")
plot.plot_accuracy_fix_agg("./results", "./fix_agg")
"""
#plot.plot_accuracy_fix_agg_best_setting("./results", "./best_plots", "./best_hyperparameters")
plot.heat_map_validation("./results", "./heat_map", "./best_hyperparameters")