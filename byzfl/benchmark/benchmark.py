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

import os
import copy
import argparse

# Global variable to keep track of training progress
counter = None

def init_pool_processes(shared_value):
    """
    Initialize a global counter variable for multiprocess tracking.

    Parameters
    ----------
    shared_value : multiprocessing.Value
        A shared memory integer used to track the number of finished trainings.
    """
    global counter
    counter = shared_value


def run_training(params):
    """
    Run a single training job, then increment the global training counter.

    Parameters
    ----------
    params : dict
        A dictionary containing all necessary parameters for the training job.
    """
    start_training(params)
    with counter.get_lock():
        print(f"Training {counter.value} done")
        counter.value += 1

"""
def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_jobs', type=int, help='Number of Jobs (multiprocessing)')
    args = parser.parse_args()

    nb_jobs = args.nb_jobs if args.nb_jobs is not None else 1
    print(f"Running {nb_jobs} experiments in parallel")
    return nb_jobs
"""

def eliminate_experiments_done(dict_list):
    """
    Remove any configurations (experiments) that have already been completed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries for each experiment.

    Returns
    -------
    list of dict
        The filtered list of configurations for which experiments are not yet done.
    """
    if not dict_list:
        return dict_list

    directory = dict_list[0]["evaluation_and_results"]["results_directory"]
    if not os.path.isdir(directory):
        return dict_list

    folders = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]

    # If there are no subfolders, no experiments are completed yet
    if not folders:
        return dict_list

    new_dict_list = []
    for setting in dict_list:
        pre_aggregation_names = [
            agg['name'] for agg in setting["pre_aggregators"]
        ]
        folder_name = (
            f"{setting['model']['dataset_name']}_"
            f"{setting['model']['name']}_"
            f"n_{setting['benchmark_config']['nb_workers']}_"
            f"f_{setting['benchmark_config']['nb_byz']}_"
            f"d_{setting['benchmark_config']['declared_nb_byz']}_"
            f"{setting['benchmark_config']['data_distribution']['name']}_"
            f"{setting['benchmark_config']['data_distribution']['distribution_parameter']}_"
            f"{setting['aggregator']['name']}_"
            f"{'_'.join(pre_aggregation_names)}_"
            f"{setting['attack']['name']}_"
            f"lr_{setting['server']['learning_rate']}_"
            f"mom_{setting['honest_nodes']['momentum']}_"
            f"wd_{setting['server']['weight_decay']}"
        )

        if folder_name in folders:
            # Check if a particular seed combination is already done
            training_seed = setting["benchmark_config"]["training_seed"]
            data_distribution_seed = setting["benchmark_config"]["data_distribution_seed"]

            file_name = (
                f"train_time_tr_seed_{training_seed}"
                f"_dd_seed_{data_distribution_seed}.txt"
            )
            if file_name not in os.listdir(os.path.join(directory, folder_name)):
                new_dict_list.append(setting)
        else:
            new_dict_list.append(setting)

    return new_dict_list


def delegate_training_seeds(dict_list):
    """
    For each configuration, generate new configurations for each specified training seed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries (each containing a base training_seed).

    Returns
    -------
    list of dict
        A new list of configurations, each with a unique training_seed.
    """
    new_dict_list = []
    for setting in dict_list:
        original_seed = setting["benchmark_config"]["training_seed"]
        nb_seeds = setting["benchmark_config"]["nb_training_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["training_seed"] = original_seed + i
            new_dict_list.append(new_setting)
    return new_dict_list


def delegate_data_distribution_seeds(dict_list):
    """
    For each configuration, generate new configurations for each specified data distribution seed.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries (each containing a base data_distribution_seed).

    Returns
    -------
    list of dict
        A new list of configurations, each with a unique data_distribution_seed.
    """
    new_dict_list = []
    for setting in dict_list:
        original_seed = setting["benchmark_config"]["data_distribution_seed"]
        nb_seeds = setting["benchmark_config"]["nb_data_distribution_seeds"]
        for i in range(nb_seeds):
            new_setting = copy.deepcopy(setting)
            new_setting["benchmark_config"]["data_distribution_seed"] = original_seed + i
            new_dict_list.append(new_setting)
    return new_dict_list


def remove_real_greater_declared(dict_list):
    """
    Filter out configurations where the real number of Byzantine workers
    exceeds the declared number.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The filtered list where declared_nb_byz >= nb_byz.
    """
    new_dict_list = []
    for setting in dict_list:
        real_byz = setting["benchmark_config"]["nb_byz"]
        declared_byz = setting["benchmark_config"]["declared_nb_byz"]
        if declared_byz >= real_byz:
            new_dict_list.append(setting)
    return new_dict_list


def remove_real_not_equal_declared(dict_list):
    """
    Filter out configurations where the real number of Byzantine workers
    is not equal to the declared number.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The filtered list where declared_nb_byz == nb_byz.
    """
    new_dict_list = []
    for setting in dict_list:
        real_byz = setting["benchmark_config"]["nb_byz"]
        declared_byz = setting["benchmark_config"]["declared_nb_byz"]
        if declared_byz == real_byz:
            new_dict_list.append(setting)
    return new_dict_list


def set_declared_as_aggregation_parameter(dict_list):
    """
    For each configuration, set the aggregator and preaggregator parameter 'f' to the declared number of Byzantine workers.

    Parameters
    ----------
    dict_list : list of dict
        A list of configuration dictionaries.

    Returns
    -------
    list of dict
        The modified list with aggregator parameters updated.
    """
    for setting in dict_list:
        declared_byz = setting["benchmark_config"]["declared_nb_byz"]
        setting["aggregator"]["parameters"]["f"] = declared_byz

        for pre_agg in setting["pre_aggregators"]:
                pre_agg["parameters"]["f"] = declared_byz
                
    return dict_list

def compute_number_of_honest_workers(dict_list):
    for data in dict_list:
        # Adjust the number of workers if needed
        if data["benchmark_config"]["fix_workers_as_honest"]:
            data["benchmark_config"]["nb_honest"] = data["benchmark_config"]["nb_workers"]
            data["benchmark_config"]["nb_workers"] = (
                data["benchmark_config"]["nb_honest"]
                + data["benchmark_config"]["nb_byz"]
            )
        else:
            data["benchmark_config"]["nb_honest"] = (
                data["benchmark_config"]["nb_workers"]
                - data["benchmark_config"]["nb_byz"]
            )
    return dict_list


def run_benchmark(nb_jobs=1):
    """
    Run benchmark experiments in parallel, based on configurations defined
    in 'config.json'.
    """
    # Attempt to load config.json or create one if not found
    try:
        with open('config.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("'config.json' not found. Creating a default one...")
        with open('config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
        print("'config.json' created successfully.")
        print("Please configure the experiment you want to run and re-run.")
        return

    # Determine the results directory (default to ./results)
    results_directory = data["evaluation_and_results"].get("results_directory", "./results")
    os.makedirs(results_directory, exist_ok=True)

    # Save the current config inside the results directory
    config_path = os.path.join(results_directory, "config.json")
    with open(config_path, 'w') as json_file:
        json.dump(data, json_file, indent=4, separators=(',', ': '))

    # Generate all combination dictionaries
    restriction_list = ["pre_aggregators", "milestones"]
    dict_list = generate_all_combinations(data, restriction_list)

    # Filter combinations based on real vs. declared Byzantines
    if data["benchmark_config"]["declared_equal_real"]:
        dict_list = remove_real_not_equal_declared(dict_list)
    else:
        dict_list = remove_real_greater_declared(dict_list)

    # Set declared parameters in the dictionaries where necessary
    dict_list = set_declared_as_aggregation_parameter(dict_list)

    # Compute the number of honest workers
    dict_list = compute_number_of_honest_workers(dict_list)

    # Assign seeds
    dict_list = delegate_training_seeds(dict_list)
    dict_list = delegate_data_distribution_seeds(dict_list)

    # Remove already completed experiments
    dict_list = eliminate_experiments_done(dict_list)

    print(f"Total trainings to do: {len(dict_list)}")
    print(f"Running {nb_jobs} trainings in parallel...")

    counter = Value('i', 0)
    with Pool(initializer=init_pool_processes, initargs=(counter,), processes=nb_jobs) as pool:
        pool.map(run_training, dict_list)

    print("All trainings finished.")