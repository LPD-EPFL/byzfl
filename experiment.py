import random
import time

import torch
import numpy as np

from train import Train
from managers import FileManager, ParamsManager

class Experiment(object):
    def __init__(self, params=None):
        self.param_manager = ParamsManager(params)

        file_manager_params = {
            "result_path": self.param_manager.get_results_directory(),
            "dataset_name": self.param_manager.get_dataset_name(),
            "model_name": self.param_manager.get_model_name(),
            "nb_workers": self.param_manager.get_nb_workers(),
            "nb_byz": self.param_manager.get_nb_byz(),
            "data_distribution_name": 
                self.param_manager.get_name_data_distribution(),
            "data_distribution_params": 
                self.param_manager.get_parameters_data_distribution(),
            "aggregation_name": self.param_manager.get_aggregator_name(),
            "pre_aggregation_names":  [
                dict['name']
                for dict in self.param_manager.get_preaggregators()
            ],
            "attack_name": self.param_manager.get_attack_name(),
            "learning_rate": self.param_manager.get_learning_rate(),
            "momentum": self.param_manager.get_momentum(),
            "weight_decay": self.param_manager.get_weight_decay(),
            "learning_rate_decay": self.param_manager.get_learning_rate_decay()
        }

        if self.param_manager.get_nb_honest() is not None:
            file_manager_params["nb_workers"] = self.param_manager.get_nb_honest() + self.param_manager.get_nb_byz()

        self.file_manager = FileManager(file_manager_params)
        self.file_manager.save_config_dict(self.param_manager.get_data())

        self.save_models = self.param_manager.get_save_models()
        self.evaluate_on_test = self.param_manager.get_evaluate_on_test()
        self.display = self.param_manager.get_display_results()

    def run(self):
        seed = self.param_manager.get_seed()
        nb_seeds = self.param_manager.get_nb_seeds()
        for _ in range(nb_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            random.seed(seed)

            start_time = time.time()

            train = Train(self.param_manager.get_flatten_info())

            train.run_SGD()

            accuracy_list = train.get_accuracy_list()

            loss_matrix = train.get_loss_list()

            if self.display:
                print("\n")
                print("Agg: "+ self.param_manager.get_aggregator_name())
                print("DataDist: "+ 
                      self.param_manager.get_name_data_distribution())
                print("Attack: " + self.param_manager.get_attack_name())
                print("Nb_byz: " + str(self.param_manager.get_nb_byz()))
                print("Seed: " + str(seed))
                accuracy_str = ", ".join(str(accuracy) for accuracy in accuracy_list)
                print("Accuracy list: " + accuracy_str)
                print("\n")
            
            end_time = time.time()

            execution_time = end_time - start_time

            self.file_manager.write_array_in_file(
                accuracy_list, 
                "train_accuracy_seed_" + str(seed) + ".txt"
            )

            self.file_manager.write_array_in_file(
                np.array(execution_time),
                "train_time_seed_" + str(seed) + ".txt"
            )

            for i, loss in enumerate(loss_matrix):
                self.file_manager.save_loss(loss, seed, i)

            if self.evaluate_on_test:
                test_accuracy_list = train.get_test_accuracy_list()
                self.file_manager.write_array_in_file(
                    test_accuracy_list, 
                    "test_accuracy_seed_" + str(seed) + ".txt"
                )

            if self.save_models:
                for i, state_dict in enumerate(train.get_model_list()):
                    self.file_manager.save_state_dict(
                        state_dict,
                        seed,
                        i * self.param_manager.get_evaluation_delta()
                    )
            seed += 1