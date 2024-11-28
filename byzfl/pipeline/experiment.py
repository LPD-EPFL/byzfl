import argparse
import json
from multiprocessing import Pool, Value
import os
import copy

from byzfl.pipeline.train import Train

default_settings = {
    "general": {
        "training_seed": 0,
        "nb_training_seeds": 5,
        "device": "cuda",
        "nb_workers": None,
        "nb_honest": 10,
        "nb_byz": [1, 3, 5, 7, 9],
        "declared_nb_byz": [1, 3, 5, 7, 9],
        "declared_equal_real": True,
        "size_train_set": 0.8,
        "nb_steps": 800,
        "evaluation_delta": 50,
        "evaluate_on_test": True,
        "store_training_accuracy": True,
        "store_training_loss": True,
        "store_models": False,
        "data_folder": None,
        "results_directory": "results"
    },
    "model": {
        "name": "cnn_mnist",
        "dataset_name": "mnist",
        "nb_labels": 10,
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1,
        "data_distribution": [
            {
                "name": "gamma_similarity_niid",
                "distribution_parameter": [1.0, 0.75, 0.5, 0.25, 0.0]
            }
        ],
        "loss": "NLLLoss"
    },
    "aggregator": [
        {
            "name": "Median",
            "parameters": {}
        },
        {
            "name": "TrMean",
            "parameters": {}
        },
        {
            "name": "GeometricMedian",
            "parameters": {
                "nu": 0.1,
                "T": 3
            }
        },
        {
            "name": "MultiKrum",
            "parameters": {}
        }
    ],
    "pre_aggregators": [
        {
            "name": "Clipping", 
            "parameters": {}
        },
        {
            "name": "NNM", 
            "parameters": {}
        }
    ],
    "server": {
        "batch_norm_momentum": None,
        "batch_size_validation": 100,
        "learning_rate_decay": 1.0,
        "milestones": []
    },
    "honest_nodes": {
        "momentum": [0.0, 0.25, 0.5, 0.75, 0.9, 0.99],
        "batch_size": 25,
        "learning_rate": [0.1, 0.25, 0.35],
        "weight_decay": 1e-4
    },
    "attack": [
        {
            "name": "SignFlipping",
            "parameters": {},
            "attack_optimizer": {
                "name": None
            }
        },
        {
            "name": "LabelFlipping",
            "parameters": {},
            "attack_optimizer": {
                "name": None
            }
        },
        {
            "name": "FallOfEmpires",
            "parameters": {},
            "attack_optimizer": {
                "name": "LineMaximize"
            }
        },
        {
            "name": "LittleIsEnough",
            "parameters": {},
            "attack_optimizer": {
                "name": "LineMaximize"
            }
        },
        {
            "name": "Mimic",
            "parameters": {},
            "attack_optimizer": {
                "name": "WorkerWithMaxVariance",
                "parameters": {
                    "steps_to_learn": 200
                }
            }
        }
    ]
}

def generate_all_combinations_aux(list_dict, orig_dict, aux_dict, rest_list):
    if len(aux_dict) < len(orig_dict):
        key = list(orig_dict)[len(aux_dict)]
        if isinstance(orig_dict[key], list):
            if not orig_dict[key] or (key in rest_list and 
                not isinstance(orig_dict[key][0], list)):
                aux_dict[key] = orig_dict[key]
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              aux_dict, 
                                              rest_list)
            else:
                for item in orig_dict[key]:
                    if isinstance(item, dict):
                        new_list_dict = []
                        new_aux_dict = {}
                        generate_all_combinations_aux(new_list_dict, 
                                                    item, 
                                                    new_aux_dict, 
                                                    rest_list)
                    else:
                        new_list_dict = [item]
                    for new_dict in new_list_dict:
                        new_aux_dict = aux_dict.copy()
                        new_aux_dict[key] = new_dict
                        
                        generate_all_combinations_aux(list_dict,
                                                    orig_dict, 
                                                    new_aux_dict, 
                                                    rest_list)
        elif isinstance(orig_dict[key], dict):
            new_list_dict = []
            new_aux_dict = {}
            generate_all_combinations_aux(new_list_dict, 
                                          orig_dict[key], 
                                          new_aux_dict, 
                                          rest_list)
            for dictionary in new_list_dict:
                new_aux_dict = aux_dict.copy()
                new_aux_dict[key] = dictionary
                generate_all_combinations_aux(list_dict, 
                                              orig_dict, 
                                              new_aux_dict, 
                                              rest_list)
        else:
            aux_dict[key] = orig_dict[key]
            generate_all_combinations_aux(list_dict, 
                                          orig_dict, 
                                          aux_dict, 
                                          rest_list)
    else:
        list_dict.append(aux_dict)

def generate_all_combinations(original_dict, restriction_list):
    list_dict = []
    aux_dict = {}
    generate_all_combinations_aux(list_dict, original_dict, aux_dict, restriction_list)
    return list_dict

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
    if len(dict_list) != 0:
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
                    + "wd_" + str(setting["honest_nodes"]["weight_decay"])
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

"""
if __name__ == '__main__':
    nb_jobs = process_args()
    data = {}
    try:
        with open('settings.json', 'r') as file:
            data = json.load(file)
    except:
        print(f"{'settings.json'} not found.")

        with open('settings.json', 'w') as f:
            json.dump(default_settings, f, indent=4)

        print(f"{'settings.json'} created successfully.")
        print("Please configure the experiment you want to run.")
        exit()
    
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
"""

def run_experiment(nb_jobs=1):
    data = {}
    try:
        with open('settings.json', 'r') as file:
            data = json.load(file)
    except:
        print(f"{'settings.json'} not found.")

        with open('settings.json', 'w') as f:
            json.dump(default_settings, f, indent=4)

        print(f"{'settings.json'} created successfully.")
        print("Please configure the experiment you want to run.")
        exit()
    
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
