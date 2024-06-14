import argparse
import json
from multiprocessing import Pool, Value
import os
import copy

from combinations import generate_all_combinations
from train import Train

def init_pool_processes(shared_value):
    global counter
    counter = shared_value

def run_training(params):
    train = Train(params)
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
                + "d_" + str(setting["general"]["declared_nb_byz"]) + "_"
                + setting["model"]["data_distribution"]["name"] + "_"
                + str(setting["model"]["data_distribution"]["distribution_parameter"]) + "_" 
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
                training_seed = setting["general"]["training_seed"]
                data_distribution_seed = setting["model"]["data_distribution_seed"]
                files = os.listdir(directory+"/"+folder_name)
                name = "train_time_tr_seed_" + str(training_seed) + "_dd_seed_" + str(data_distribution_seed) + ".txt"
                if not name in files:
                    real_dict_list.append(setting)
            else:
                real_dict_list.append(setting)
        return real_dict_list
    else:
        return dict_list

def delegate_training_seeds(dict_list):
    real_dict_list = []
    for setting in dict_list:
        original_seed = setting["general"]["training_seed"]
        nb_seeds = setting["general"]["nb_training_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["general"]["training_seed"] = original_seed + i
            real_dict_list.append(new_setting)
    return real_dict_list

def delegate_data_distribution_seeds(dict_list):
    real_dict_list = []
    for setting in dict_list:
        original_seed = setting["model"]["data_distribution_seed"]
        nb_seeds = setting["model"]["nb_data_distribution_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["model"]["data_distribution_seed"] = original_seed + i
            real_dict_list.append(new_setting)
    return real_dict_list

def remove_real_greater_declared(dict_list):
    real_dict_list = []
    for setting in dict_list:
        if setting["general"]["declared_nb_byz"] >= setting["general"]["nb_byz"]:
            real_dict_list.append(setting)
    return real_dict_list

def remove_real_not_equal_declared(dict_list):
    real_dict_list = []
    for setting in dict_list:
        if setting["general"]["declared_nb_byz"] == setting["general"]["nb_byz"]:
            real_dict_list.append(setting)
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

    if data["general"]["declared_equal_real"]:
        dict_list = remove_real_not_equal_declared(dict_list)
    else:
        dict_list = remove_real_greater_declared(dict_list)

    #Do a setting for every seed
    dict_list = delegate_training_seeds(dict_list)
    dict_list = delegate_data_distribution_seeds(dict_list)

    #Do only experiments that haven't been done
    dict_list = eliminate_experiments_done(dict_list)

    print("Total trainings to do: " + str(len(dict_list)))

    counter = Value('i', 0)
    with Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs) as pool:
        pool.map(run_training, dict_list)