import argparse
import json
import multiprocessing
import os
import math
import copy

import numpy as np

from combinations import generate_all_combinations
from experiment import Experiment

def run_experiment(params):
    exp = Experiment(params=params)
    exp.run()

def generate_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch(batch):
    with multiprocessing.Pool() as pool:
        pool.map(run_experiment, batch)

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_jobs', type=int, help='Number of Jobs (multiprocessing)')

    args = parser.parse_args()
    
    nb_jobs = 1
    if args.nb_jobs is not None:
        nb_jobs = args.nb_jobs
    print("Running " + str(nb_jobs) + " experiments in parallel")
    return nb_jobs

def eliminate_experiments_done(dict_list):
    directory = dict_list[0]["general"]["results_directory"]
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    if len(folders) != 0:
        real_dict_list = []
        for setting in dict_list:
            if setting["general"]["nb_workers"] == None:
                setting["general"]["nb_workers"] = setting["general"]["nb_honest"] + setting["general"]["nb_byz"]
            # First check folder
            pre_aggregation_names =  [
                dict['name']
                for dict in setting["pre_aggregators"]
            ]
            folder_name = str(
                setting["model"]["dataset_name"] + "_" 
                + setting["model"]["name"] + "_" 
                +"n_" + str(setting["general"]["nb_workers"]) + "_" 
                + "f_" + str(setting["general"]["nb_byz"]) + "_" 
                + setting["model"]["data_distribution"]["name"] + "_"
                "_".join([
                    str(valor) 
                    for valor in setting["model"]["data_distribution"]["parameters"].values()
                ]) + "_" 
                + setting["aggregator"]["name"] + "_"
                + "_".join(pre_aggregation_names) + "_"
                + setting["attack"]["name"] + "_" 
                + "lr_" + str(setting["honest_nodes"]["learning_rate"]) + "_" 
                + "mom_" + str(setting["honest_nodes"]["momentum"]) + "_" 
                + "wd_" + str(setting["honest_nodes"]["weight_decay"]) + "_"
                + "lr_decay_" + str(setting["server"]["learning_rate_decay"])
            )

            if folder_name in folders:
                #Now we check the seeds
                seed = setting["general"]["seed"]
                nb_seeds = setting["general"]["nb_seeds"]
                files = os.listdir(directory+"/"+folder_name)
                for i in range(nb_seeds):
                    name = "train_accuracy_seed_" + str(seed+i) + ".txt"
                    if not name in files:
                        setting["general"]["seed"] = seed+i
                        setting["general"]["nb_seeds"] = nb_seeds-i
                        real_dict_list.append(setting)
                        break
            else:
                real_dict_list.append(setting)
        return real_dict_list
    else:
        return dict_list

def train_best_setting(setting, path_best_hyperparameters):
    if setting["general"]["nb_workers"] is None:
        nb_workers = setting["general"]["nb_honest"] + setting["general"]["nb_byz"]
    if len(setting["model"]["data_distribution"]["parameters"]) > 0:
        parameters = setting["model"]["data_distribution"]["parameters"]
        factor = str(parameters[list(parameters.keys())[0]])
    else:
        factor = ""

    file_name = str(
        setting["model"]["dataset_name"] + "_"
        + setting["model"]["name"] + "_n_"
        + str(nb_workers) + "_f_"
        + str(setting["general"]["nb_byz"]) + "_"
        + setting["model"]["data_distribution"]["name"]
        + factor + "_"
        + "_".join(pre_agg["name"] for pre_agg in setting["pre_aggregators"]) + "_"
        + setting["aggregator"]["name"] + ".txt"
    )

    steps_file_name = str(
        setting["model"]["dataset_name"] + "_"
        + setting["model"]["name"] + "_n_"
        + str(nb_workers) + "_f_"
        + str(setting["general"]["nb_byz"]) + "_"
        + setting["model"]["data_distribution"]["name"]
        + factor + "_"
        + "_".join(pre_agg["name"] for pre_agg in setting["pre_aggregators"]) + "_"
        + setting["aggregator"]["name"] + "_"
        + setting["attack"]["name"]
        + ".txt"
    )

    if os.path.exists(path_best_hyperparameters +"/hyperparameters/"+ file_name) and \
        os.path.exists(path_best_hyperparameters +"/better_step/"+ steps_file_name):

        best_hyperparameters = np.loadtxt(path_best_hyperparameters +"/hyperparameters/"+ file_name)
        steps = int(np.loadtxt(path_best_hyperparameters +"/better_step/"+ steps_file_name))

        lr = best_hyperparameters[0]
        momentum = best_hyperparameters[1]
        wd = best_hyperparameters[2]

        new_setting = copy.deepcopy(setting)

        new_setting["honest_nodes"]["learning_rate"] = lr
        new_setting["honest_nodes"]["momentum"] = momentum
        new_setting["honest_nodes"]["weight_decay"] = wd
        new_setting["general"]["nb_steps"] = steps

        return new_setting
    else:
        return setting

def remove_duplicates(dict_list):
    set = {json.dumps(setting, sort_keys=True) for setting in dict_list}
    return [json.loads(unique_setting) for unique_setting in set]

if __name__ == '__main__':
    nb_jobs = process_args()
    data = {}
    try:
        with open('settings.json', 'r') as file:
            data = json.load(file)
    except:
        print("ERROR")
    
    results_directory = None
    if data["general"]["results_directory"] is None:
        results_directory = "./results"
    else:
        results_directory = data["general"]["results_directory"]

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    with open(results_directory+"/settings.json", 'w') as json_file:
            json.dump(data, json_file, indent=4, separators=(',', ': '))

    restriction_list = ["pre_aggregators", "milestones"]
    dict_list = generate_all_combinations(data, restriction_list)
    optimized_dict_list = []
    
    #Best setting found before
    for setting in dict_list:
        optimized_setting = train_best_setting(setting, "./best_hyperparameters")
        optimized_dict_list.append(optimized_setting)
    
    dict_list = remove_duplicates(optimized_dict_list)
    
    #Do only experiments that haven't been done
    dict_list = eliminate_experiments_done(dict_list)

    total_batches = str(math.ceil(len(dict_list)/nb_jobs))
    print("Total experiments: " + str(len(dict_list)))
    print("Total batches: " + total_batches)
    batches = generate_batches(dict_list, nb_jobs)
    i = 0
    for batch in batches:
        process_batch(batch)
        print("Batch "+ str(i)+ " done")
        i += 1