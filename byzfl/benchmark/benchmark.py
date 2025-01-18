import argparse
import json
from multiprocessing import Pool, Value
import os
import copy

from byzfl.benchmark.train import start_training

default_config = {
    "benchmark_config": {
        "device": "cuda",
        "training_seed": 0,
        "nb_training_seeds": 5,
        "nb_workers": 5,
        "nb_byz": 1,
        "declared_nb_byz": 1,
        "declared_equal_real": True,
        "fix_workers_as_honest": False,
        "size_train_set": 0.8,
        "data_distribution_seed": 0,
        "nb_data_distribution_seeds": 1,
        "data_distribution": [
            {
                "name": "gamma_similarity_niid",
                "distribution_parameter": 0.5
            }
        ],
    },
    "model": {
        "name": "cnn_mnist",
        "dataset_name": "mnist",
        "nb_labels": 10,
        "loss": "NLLLoss"
    },
    "aggregator": [
        {
            "name": "TrMean",
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
        "learning_rate": 0.25,
        "weight_decay": 1e-4,
        "nb_steps": 800,
        "batch_norm_momentum": None,
        "batch_size_validation": 100,
        "learning_rate_decay": 1.0,
        "milestones": []
    },
    "honest_nodes": {
        "momentum": 0.9,
        "batch_size": 25
    },
    "attack": [
        {
            "name": "SignFlipping",
            "parameters": {},
            "attack_optimizer": {
                "name": None
            }
        }
    ],
    "evaluation_and_results": {
        "evaluation_delta": 50,
        "evaluate_on_test": True,
        "store_training_accuracy": True,
        "store_training_loss": True,
        "store_models": False,
        "data_folder": None,
        "results_directory": "results"
    },
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
    start_training(params)
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
        directory = dict_list[0]["evaluation_and_results"]["results_directory"]
        folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        if len(folders) != 0:
            real_dict_list = []
            for setting in dict_list:
                pre_aggregation_names =  [
                    dict['name']
                    for dict in setting["pre_aggregators"]
                ]
                folder_name = str(
                    setting["model"]["dataset_name"] + "_" 
                    + setting["model"]["name"] + "_" 
                    +"n_" + str(setting["benchmark_config"]["nb_workers"]) + "_" 
                    + "f_" + str(setting["benchmark_config"]["nb_byz"]) + "_" 
                    + "d_" + str(setting["benchmark_config"]["declared_nb_byz"]) + "_"
                    + setting["benchmark_config"]["data_distribution"]["name"] + "_"
                    + str(setting["benchmark_config"]["data_distribution"]["distribution_parameter"]) + "_" 
                    + setting["aggregator"]["name"] + "_"
                    + "_".join(pre_aggregation_names) + "_"
                    + setting["attack"]["name"] + "_" 
                    + "lr_" + str(setting["server"]["learning_rate"]) + "_" 
                    + "mom_" + str(setting["honest_nodes"]["momentum"]) + "_" 
                    + "wd_" + str(setting["server"]["weight_decay"])
                )

                if folder_name in folders:
                    #Now we check the seeds
                    training_seed = setting["benchmark_config"]["training_seed"]
                    data_distribution_seed = setting["benchmark_config"]["data_distribution_seed"]
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
        original_seed = setting["benchmark_config"]["training_seed"]
        nb_seeds = setting["benchmark_config"]["nb_training_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["training_seed"] = original_seed + i
            real_dict_list.append(new_setting)
    return real_dict_list

def delegate_data_distribution_seeds(dict_list):
    real_dict_list = []
    for setting in dict_list:
        original_seed = setting["benchmark_config"]["data_distribution_seed"]
        nb_seeds = setting["benchmark_config"]["nb_data_distribution_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["data_distribution_seed"] = original_seed + i
            real_dict_list.append(new_setting)
    return real_dict_list

def remove_real_greater_declared(dict_list):
    real_dict_list = []
    for setting in dict_list:
        if setting["benchmark_config"]["declared_nb_byz"] >= setting["benchmark_config"]["nb_byz"]:
            real_dict_list.append(setting)
    return real_dict_list

def remove_real_not_equal_declared(dict_list):
    real_dict_list = []
    for setting in dict_list:
        if setting["benchmark_config"]["declared_nb_byz"] == setting["benchmark_config"]["nb_byz"]:
            real_dict_list.append(setting)
    return real_dict_list

def run_benchmark(nb_jobs=1):
    data = {}
    try:
        with open('config.json', 'r') as file:
            data = json.load(file)
    except:
        print(f"{'config.json'} not found.")

        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=4)

        print(f"{'config.json'} created successfully.")
        print("Please configure the experiment you want to run.")
        exit()
    
    results_directory = None
    if data["evaluation_and_results"]["results_directory"] is None:
        results_directory = "./results"
    else:
        results_directory = data["evaluation_and_results"]["results_directory"]

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    with open(results_directory+"/config.json", 'w') as json_file:
            json.dump(data, json_file, indent=4, separators=(',', ': '))
    
    if data["benchmark_config"]["fix_workers_as_honest"]:
        data["benchmark_config"]["nb_honest"] = data["benchmark_config"]["nb_workers"]
        data["benchmark_config"]["nb_workers"] = data["benchmark_config"]["nb_honest"] \
            + data["benchmark_config"]["nb_byz"]
    else:
        data["benchmark_config"]["nb_honest"] = data["benchmark_config"]["nb_workers"] \
            - data["benchmark_config"]["nb_byz"]
        

    restriction_list = ["pre_aggregators", "milestones"]
    dict_list = generate_all_combinations(data, restriction_list)

    if data["benchmark_config"]["declared_equal_real"]:
        dict_list = remove_real_not_equal_declared(dict_list)
    else:
        dict_list = remove_real_greater_declared(dict_list)

    #Do a setting for every seed
    dict_list = delegate_training_seeds(dict_list)
    dict_list = delegate_data_distribution_seeds(dict_list)

    #Do only experiments that haven't been done
    dict_list = eliminate_experiments_done(dict_list)

    print("Total trainings to do: " + str(len(dict_list)))

    print(f"Running {nb_jobs} trainings in parallel")

    counter = Value('i', 0)
    with Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs) as pool:
        pool.map(run_training, dict_list)

    print("Trainings finished")