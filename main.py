import argparse
import json
from multiprocessing import Pool, Value
import os
import copy

import numpy as np

from combinations import generate_all_combinations
from train import Train
from managers import ParamsManager

def init_pool_processes(shared_value):
    global counter
    counter = shared_value

def run_training(params):
    param_manager = ParamsManager(params=params)
    train = Train(param_manager.get_flatten_info(), param_manager.get_data())
    train.run_SGD()
    with counter.get_lock():
        print("Training " + str(counter.value) + " done")
        counter.value += 1

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
                files = os.listdir(directory+"/"+folder_name)
                name = "train_accuracy_seed_" + str(seed) + ".txt"
                if not name in files:
                    real_dict_list.append(setting)
            else:
                real_dict_list.append(setting)
        return real_dict_list
    else:
        return dict_list

def train_best_setting(setting, path_best_hyperparameters):
    if setting["general"]["nb_workers"] is None:
        setting["general"]["nb_workers"] = setting["general"]["nb_honest"] + setting["general"]["nb_byz"]
    if len(setting["model"]["data_distribution"]["parameters"]) > 0:
        parameters = setting["model"]["data_distribution"]["parameters"]
        factor = str(parameters[list(parameters.keys())[0]])
    else:
        factor = ""

    file_name = str(
        setting["model"]["dataset_name"] + "_"
        + setting["model"]["name"] + "_n_"
        + str(setting["general"]["nb_workers"]) + "_f_"
        + str(setting["general"]["nb_byz"]) + "_"
        + setting["model"]["data_distribution"]["name"]
        + factor + "_"
        + "_".join(pre_agg["name"] for pre_agg in setting["pre_aggregators"]) + "_"
        + setting["aggregator"]["name"] + ".txt"
    )
    
    """
    steps_file_name = str(
        setting["model"]["dataset_name"] + "_"
        + setting["model"]["name"] + "_n_"
        + str(setting["general"]["nb_workers"]) + "_f_"
        + str(setting["general"]["nb_byz"]) + "_"
        + setting["model"]["data_distribution"]["name"]
        + factor + "_"
        + "_".join(pre_agg["name"] for pre_agg in setting["pre_aggregators"]) + "_"
        + setting["aggregator"]["name"] + "_"
        + setting["attack"]["name"]
        + ".txt"
    )
    """

    if os.path.exists(path_best_hyperparameters +"/hyperparameters/"+ file_name):

        best_hyperparameters = np.loadtxt(path_best_hyperparameters +"/hyperparameters/"+ file_name)
        #steps = int(np.loadtxt(path_best_hyperparameters +"/better_step/"+ steps_file_name))

        lr = best_hyperparameters[0]
        momentum = best_hyperparameters[1]
        wd = best_hyperparameters[2]

        new_setting = copy.deepcopy(setting)

        new_setting["honest_nodes"]["learning_rate"] = lr
        new_setting["honest_nodes"]["momentum"] = momentum
        new_setting["honest_nodes"]["weight_decay"] = wd
        #new_setting["general"]["nb_steps"] = steps

        return new_setting
    else:
        return setting

def remove_duplicates(dict_list):
    set = {json.dumps(setting, sort_keys=True) for setting in dict_list}
    return [json.loads(unique_setting) for unique_setting in set]

def delegate_seeds(dict_list):
    real_dict_list = []
    for setting in dict_list:
        original_seed = setting["general"]["seed"]
        nb_seeds = setting["general"]["nb_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["general"]["original_seed"] = original_seed
            new_setting["general"]["seed"] = original_seed + i
            real_dict_list.append(new_setting)
    return real_dict_list

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

    #Do a setting for every seed
    dict_list = delegate_seeds(dict_list)

    #Do only experiments that haven't been done
    dict_list = eliminate_experiments_done(dict_list)

    print("Total trainings to do: " + str(len(dict_list)))

    counter = Value('i', 0)
    with Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs) as pool:
        pool.map(run_training, dict_list)