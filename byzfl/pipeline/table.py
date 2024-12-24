import math
import json
import os

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class Table(object):

    def custom_dict_to_str(self, d):
        if not d:
            return ''
        else:
            return str(d)
    
    def plot_table(self, path_to_results, path_to_tables):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ e)

        if not os.path.exists(path_to_tables):
            os.makedirs(path_to_tables)
        
        seed = data["general"]["seed"]
        nb_seeds = data["general"]["nb_seeds"]
        nb_workers = data["general"]["nb_workers"]
        nb_byz = data["general"]["nb_byz"]
        nb_steps = data["general"]["nb_steps"] #Fixed
        evaluation_delta = data["general"]["evaluation_delta"] #Fixed
        bit_precision = data["general"]["bit_precision"]

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
        
        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))

        for nb_nodes in nb_workers:
            for nb_byzantine in nb_byz:
                for lr in lr_list:
                    for momentum in momentum_list:
                        for wd in wd_list:
                            for data_dist in data_distributions:
                                alpha = ""
                                if data_dist["name"] == "dirichlet_niid":
                                    alpha = data_dist["parameters"]["alpha"]
                                
                                big_table = np.zeros((len(aggregators), len(attacks)), dtype=np.dtype('float,float'))

                                for x, agg in enumerate(aggregators):
                                    for pre_agg in pre_aggregators:
                                        if pre_agg is not None:
                                            pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                                            pre_agg_names = "_".join(pre_agg_list_names)
                                        else:
                                            pre_agg_names = ""
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
                                                                + str(wd))
                                                tab_acc[i][run] = genfromtxt(path_to_results+"/"+common_name+"/train_accuracy_seed_"+str(run+seed)+".txt", delimiter=',')
                                        
                                        err = np.zeros((len(attacks), nb_accuracies))
                                        for i in range(len(err)):
                                            err[i] = (1.96*np.std(tab_acc[i], axis = 0))/math.sqrt(nb_seeds)
                                        err = np.round(err, decimals=bit_precision)
                                        
                                        for i, attack in enumerate(attacks):
                                            acc_mean = np.mean(tab_acc[i], axis=0)
                                            idx_max = np.argmax(acc_mean)
                                            acc_mean = np.round(acc_mean, decimals=bit_precision)
                                            big_table[x][i] = (acc_mean[idx_max], err[i][idx_max])
                                    
                                rows_names = [agg['name'] for agg in aggregators]
                                columns_names = [attack["name"] for attack in attacks]

                                table_name = str(dataset_name + "_"
                                                 + model_name + "_n_"
                                                 + str(nb_nodes) + "_f_"
                                                 + str(nb_byzantine) + "_"
                                                 + self.custom_dict_to_str(data_dist["name"])
                                                 + str(alpha) + "_"
                                                 + self.custom_dict_to_str(agg["name"]) + "_"
                                                 + pre_agg_names + "_lr_"
                                                 + str(lr) + "_mom_"
                                                 + str(momentum) + "_wd_"
                                                 + str(wd))
                                
                                pdf_pages = PdfPages(path_to_tables+"/"+table_name + ".pdf")

                                plt.figure(figsize=(8, 6))
                                plt.axis('off')

                                plt.table(cellText=big_table,
                                        rowLabels=rows_names,
                                        colLabels=columns_names,
                                        cellLoc= 'center',
                                        loc='center',
                                        bbox=[0.15, 0.15, 0.9, 0.9])

                                pdf_pages.savefig()
                                pdf_pages.close()

    def plot_paper(self, path_to_results, path_to_tables, path_to_hyperparameters):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_tables):
            os.makedirs(path_to_tables)
        
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
                            max_worst_case_agg = np.zeros((len(aggregators), len(attacks)))
                            real_errors = np.zeros((len(aggregators), len(attacks)))
                            for k, agg in enumerate(aggregators):
                                tab_acc = np.zeros((
                                    len(attacks), 
                                    nb_seeds, 
                                    nb_accuracies
                                ))
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
                                lr_d = 1.0
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
                                err = np.round(err, decimals=bit_precision)
                                
                                for i, attack in enumerate(attacks):
                                    acc_mean = np.mean(tab_acc[i], axis=0)
                                    idx_max = np.argmax(acc_mean)
                                    acc_mean = np.round(acc_mean, decimals=bit_precision)
                                    max_worst_case_agg[k][i] = acc_mean[idx_max]
                                    real_errors[k][i] = err[i][idx_max]

                                if not os.path.exists(path_to_hyperparameters):
                                    try:
                                        os.makedirs(path_to_hyperparameters)
                                    except OSError as error:
                                        print(f"Error creating directory: {error}")
                                
                            real_table = np.zeros((len(aggregators) + 3,len(attacks) + 1), dtype=np.dtype('float,float'))
                            row_min_values = np.array([np.inf]*len(aggregators))
                            row_min_error = np.array([np.inf]*len(aggregators))
                            column_min_values = np.array([np.inf]*len(attacks))
                            column_min_error = np.array([np.inf]*len(attacks))
                            column_max_values = np.zeros((len(attacks)))
                            column_max_error = np.zeros((len(attacks)))
                            column_avg_values = list(zip(np.mean(max_worst_case_agg, axis=0), [0.0]*(len(np.mean(max_worst_case_agg, axis=0)))))
                            
                            for i in range(len(max_worst_case_agg)):
                                for j in range(len(max_worst_case_agg[0])):
                                    real_table[i][j] = (max_worst_case_agg[i][j], real_errors[i][j])
                                    if max_worst_case_agg[i][j] < row_min_values[i]:
                                        row_min_values[i] = max_worst_case_agg[i][j]
                                        row_min_error[i] = real_errors[i][j]
                                    if max_worst_case_agg[i][j] < column_min_values[j]:
                                        column_min_values[j] = max_worst_case_agg[i][j]
                                        column_min_error[j] = real_errors[i][j]
                                    if max_worst_case_agg[i][j] > column_max_values[j]:
                                        column_max_values[j] = max_worst_case_agg[i][j]
                                        column_max_error[j] = real_errors[i][j]

                            for i in range(len(row_min_values)):
                                real_table[i][len(real_table[0])-1] = (row_min_values[i], row_min_error[i])
                            
                            for i in range(len(attacks)):
                                real_table[len(real_table)-3][i] = (column_min_values[i], column_min_error[i])
                                real_table[len(real_table)-2][i] = (column_max_values[i], column_max_error[i])
                                real_table[len(real_table)-1][i] = column_avg_values[i]

                            rows_names = [agg['name'] for agg in aggregators]
                            columns_names = [attack["name"] for attack in attacks]

                            rows_names.append("Min")
                            rows_names.append("Max")
                            rows_names.append("Average")
                            columns_names.append("Min")

                            table_name = str(dataset_name + "_"
                                            + model_name + "_n_"
                                            + str(nb_nodes) + "_f_"
                                            + str(nb_byzantine) + "_"
                                            + self.custom_dict_to_str(data_dist["name"])
                                            + str(alpha) + "_"
                                            + pre_agg_names)
                            
                            pdf_pages = PdfPages(path_to_tables + "/" + table_name + ".pdf")

                            plt.figure(figsize=(8, 6))
                            plt.axis('off')

                            plt.table(cellText=real_table,
                                    rowLabels=rows_names,
                                    colLabels=columns_names,
                                    cellLoc= 'center',
                                    loc='center',
                                    bbox=[0.15, 0.15, 0.9, 0.9])
                            

                            pdf_pages.savefig()
                            pdf_pages.close()

                            plt.close()

    def best_agg_for_pair_hetero_byz(self, path_to_results, path_to_table, path_to_hyperparameters):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_table):
            os.makedirs(path_to_table)
        
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
        lr_d = 0.1
        nb_accuracies = int(1+math.ceil(nb_steps/evaluation_delta))
        for data_dist in data_distributions:
            alpha_list = [""]
            if data_dist["name"] == "dirichlet_niid":
                alpha_list = data_dist["parameters"]["alpha"]
            if data_dist["name"] == "gamma_similarity_niid":
                alpha_list = data_dist["parameters"]["gamma"]
            for pre_agg in pre_aggregators:
                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                pre_agg_names = "_".join(pre_agg_list_names)
                cube_acc_table = np.zeros((len(aggregators), len(alpha_list), len(nb_byz)))
                cube_err_table = np.zeros((len(aggregators), len(alpha_list), len(nb_byz)))
                for a, agg in enumerate(aggregators):
                    #try:
                    acc_table = np.zeros((len(alpha_list), len(nb_byz)))
                    err_table = np.zeros((len(alpha_list), len(nb_byz)))
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
                                    
                                    err = np.zeros((nb_accuracies))
                                    err = (1.96*np.std(tab_acc, axis = 0))/math.sqrt(nb_seeds)

                                    tab_acc = tab_acc.mean(axis=0)
                                    idx_accuracy = np.argmax(tab_acc)
                                    if tab_acc[idx_accuracy] < worst_accuracy:
                                        worst_accuracy = tab_acc[idx_accuracy]
                                        error_linked = err[idx_accuracy]

                                acc_table[len(acc_table)-1-x][y] = worst_accuracy
                                err_table[len(acc_table)-1-x][y] = error_linked

                    cube_acc_table[a] = acc_table
                    cube_err_table[a] = err_table
                
                max_length = -1
                for agg in aggregators:
                    aux = len(agg["name"])
                    if aux > max_length:
                        max_length = aux
                    

                idx_better_agg = np.argmax(cube_acc_table, axis=0)
                matrix = [[0 for _ in range(len(alpha_list))] for _ in range(len(nb_byz))]
                for i in range(len(alpha_list)):
                    for j in range(len(nb_byz)):
                        best_accuracy = cube_acc_table[idx_better_agg[i][j]][i][j]
                        if best_accuracy < 0.3:
                            matrix[i][j] = "No Robust"
                        else:
                            best_agg_list = [(aggregators[idx_better_agg[i][j]]["name"] + "_" + str(np.round(best_accuracy, 4)) + "_" + str(np.round(cube_err_table[idx_better_agg[i][j]][i][j], 4)))]
                            minimum_threshold = best_accuracy - cube_err_table[idx_better_agg[i][j]][i][j]
                            for k in range(len(aggregators)):
                                if k != idx_better_agg[i][j] :
                                    other_acc_confidence = cube_acc_table[k][i][j] + cube_err_table[k][i][j]
                                    if other_acc_confidence >= minimum_threshold:
                                        best_agg_list.append((aggregators[k]["name"] + "_" + str(np.round(cube_acc_table[k][i][j], 4)) + "_" + str(np.round(cube_err_table[k][i][j], 4))))
                                    
                            matrix[i][j] = "\n".join(best_agg_list)

                file_name = str(
                    dataset_name + "_"
                    + model_name + "_"
                    + self.custom_dict_to_str(data_dist["name"]) + "_"
                    + pre_agg_names + "_"
                    + "best_aggregation"
                )
                
                column_names = [str(alpha) for alpha in alpha_list]
                row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                column_names.reverse()
                
                pdf_pages = PdfPages(path_to_table + "/" + file_name + ".pdf")

                plt.figure(figsize=(8, 6))
                plt.axis('off')

                plt.table(cellText=matrix,
                        rowLabels=column_names,
                        colLabels=row_names,
                        cellLoc= 'center',
                        loc='center',
                        bbox=[0.1, 0.15, 0.9, 0.9]).scale(1, 1)
                
                #plt.rcParams.update({'font.size': 36})
                

                pdf_pages.savefig()
                pdf_pages.close()

                plt.close()
                    #except Exception as e:
                    #    print(e)
    
    def best_hyperparameters_for_pair_hetero_byz(self, path_to_results, path_to_table, path_to_hyperparameters):
        try:
            with open(path_to_results+'/settings.json', 'r') as file:
                data = json.load(file)
        except Exception as e:
            print("ERROR: "+ str(e))

        if not os.path.exists(path_to_table):
            os.makedirs(path_to_table)
        
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

        for data_dist in data_distributions:
            alpha_list = [""]
            if data_dist["name"] == "dirichlet_niid":
                alpha_list = data_dist["parameters"]["alpha"]
            if data_dist["name"] == "gamma_similarity_niid":
                alpha_list = data_dist["parameters"]["gamma"]
            for pre_agg in pre_aggregators:
                pre_agg_list_names = [one_pre_agg['name'] for one_pre_agg in pre_agg]
                pre_agg_names = "_".join(pre_agg_list_names)
                for agg in aggregators:
                    #try:
                    hyp_table = np.zeros((len(alpha_list), len(nb_byz), 2))
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

                                hyp_table[len(hyp_table)-1-x][y][0] = lr
                                hyp_table[len(hyp_table)-1-x][y][1] = momentum

                    matrix = [[0 for _ in range(len(alpha_list))] for _ in range(len(nb_byz))]
                    for i in range(len(alpha_list)):
                        for j in range(len(nb_byz)):
                            matrix[i][j] = (hyp_table[i][j][0], hyp_table[i][j][1])

                    file_name = str(
                        dataset_name + "_"
                        + model_name + "_"
                        + self.custom_dict_to_str(data_dist["name"]) + "_"
                        + pre_agg_names + "_"
                        + self.custom_dict_to_str(agg["name"])
                    )
                    
                    column_names = [str(alpha) for alpha in alpha_list]
                    row_names = [str(nb_byzantine) for nb_byzantine in nb_byz]
                    column_names.reverse()
                    
                    pdf_pages = PdfPages(path_to_table + "/" + file_name + ".pdf")

                    plt.figure(figsize=(8, 6))
                    plt.axis('off')

                    plt.table(cellText=matrix,
                            rowLabels=column_names,
                            colLabels=row_names,
                            cellLoc= 'center',
                            loc='center',
                            bbox=[0.15, 0.15, 0.9, 0.9])
                    

                    pdf_pages.savefig()
                    pdf_pages.close()

                    plt.close()
                        #except Exception as e:
                        #    print(e)


table = Table()
#table.best_hyperparameters_for_pair_hetero_byz("./results", "./best_hyperparameters_table", "./best_hyperparameters")
table.best_agg_for_pair_hetero_byz("./results", "./best_aggregations", "./best_hyperparameters")