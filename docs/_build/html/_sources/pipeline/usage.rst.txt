usage
============

Framework for Byzantine Machine Learning
----------------------------------------
This tool facilitates testing aggregations and attacks by simulating distributed machine learning environments in a fully configurable manner via the `settings.json` file.

Requirements
------------
1. Install the dependencies listed in `requirements.txt`.
2. Set the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Usage or Pipeline
-----------------
0. If this is your first time running the framework, execute `python main.py`. This will create a default `settings.json` file, which you can customize according to your requirements.
1. Configure the experiment you wish to run in the `settings.json` file.
2. Run the framework with `python main.py --nb_jobs n`, where `n` represents the number of trainings to be conducted in parallel.
3. When execution completes, run `python evaluate_results.py` to select the best hyperparameters and create heatmaps with the results.

Configuring `settings.json`
---------------------------
- **General**:
    - `training_seed`: Starting seed for training across different seeds.
    - `nb_training_seeds`: Number of seeds to be used starting from `training_seed`.
    - `device`: Device in PyTorch used for tensor computations.
    - `nb_workers`: Number of workers for training. If `null`, it defaults to `nb_honest + nb_byz`.
    - `nb_honest`: Number of honest clients.
    - `nb_byz`: Number of real Byzantine clients.
    - `declared_nb_byz`: Number of Byzantine clients that the server protects against.
    - `declared_equal_real`: Boolean that specifies whether declared Byzantine clients equal real ones.
    - `size_train_set`: Proportion of the training set used for training; remaining is used for validation.
    - `nb_steps`: Number of iterations for the learning algorithm.
    - `evaluation_delta`: Number of steps between evaluations on the validation/test set.
    - `evaluate_on_test`: Boolean for computing test accuracy.
    - `store_training_accuracy`: Boolean to store training accuracies.
    - `store_training_loss`: Boolean to store training losses.
    - `store_models`: Boolean to save PyTorch model states at each evaluation delta.
    - `batch_size_validation`: Batch size for validation/test datasets (not training).
    - `data_folder`: Path for dataset storage.
    - `results_directory`: Path for storing training information.

- **Model**:
    - `name`: Name of the model for training (must match a class in **models.py**).
    - `dataset_name`: Name of the dataset for training (defined in **dataset.py**).
    - `nb_labels`: Number of unique targets in the dataset.
    - `data_distribution_seed`: Seed for data distribution across nodes.
    - `nb_data_distribution_seeds`: Number of seeds starting from `data_distribution_seed`.
    - `data_distribution`:
        - `name`: Name of the data distribution method.
        - `distribution_parameter`: Float parameter for the distribution.
    - `loss`: PyTorch training loss.

- **Aggregator**:
    - `name`: Aggregation used by the server (defined in **aggregators.py**).
    - `parameters`: Dictionary of parameters for the aggregation.

- **PreAggregators**: List of pre-aggregations in order as they should be applied.
    - `name`: Name of the pre-aggregation used by the server (defined in **preaggregators.py**).
    - `parameters`: Dictionary of parameters for pre-aggregations.

- **Server**:
    - `batch_norm_momentum`: Momentum for federated batch normalization.
    - `learning_rate_decay`: Factor for learning rate decay at each milestone.
    - `milestones`: Steps at which learning rate decay occurs.

- **Honest Nodes**:
    - `momentum`: Momentum for the training algorithm.
    - `batch_size`: Batch size for the SGD algorithm.
    - `learning_rate`: Learning rate value.
    - `weight_decay`: Weight decay to prevent overfitting.

- **Attack**:
    - `name`: Attack name used by the server (defined in **attacks.py**).
    - `parameters`: Dictionary of parameters for the attack.
    - `attack_optimizer`:
        - `name`: Name of the optimizer for attack adjustment.
        - `parameters`: Dictionary of optimizer parameters.

Configuring `settings.json` for Multiple Settings
-------------------------------------------------
This library allows simultaneous runs of multiple settings, ideal for exploring different aggregation and attack configurations. To enable this, use a list in `settings.json` instead of a single element. Example:

    distribution_parameter: 1.0  ->  distribution_parameter: [1.0, 0.5, 0.0]


Default settings.json
--------------------------------------

This `settings.json` file is the default configuration for the framework. If it is not present, run `python main.py`, and the file will be generated automatically. You should then configure it to specify the experiment you wish to run.

.. code-block:: json

    {
        "general": {
            "training_seed": 0,
            "nb_training_seeds": 5,
            "device": "cuda",
            "nb_workers": null,
            "nb_honest": 10,
            "nb_byz": [1, 3, 5, 7, 9],
            "declared_nb_byz": [1, 3, 5, 7, 9],
            "declared_equal_real": true,
            "size_train_set": 0.8,
            "nb_steps": 800,
            "evaluation_delta": 50,
            "evaluate_on_test": true,
            "store_training_accuracy": true,
            "store_training_loss": true,
            "store_models": false,
            "data_folder": null,
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
            "batch_norm_momentum": null,
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
                    "name": null
                }
            },
            {
                "name": "LabelFlipping",
                "parameters": {},
                "attack_optimizer": {
                    "name": null
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

