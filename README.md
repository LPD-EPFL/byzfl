# Framework for Byzantine ML
This tool facilitates the testing of aggregations and attacks by simulating distributed machine learning environments in a fully configurable manner via the `settings.json` file.

# Requirements
1. Install the dependencies listed in `requirements.txt`.
2. Set the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`

# Usage or Pipeline
0. If this is your first time running the framework, execute `python main.py`. This will create a default settings.json file, which you can customize according to your requirements.
1. Configure the experiment you wish to run in the `settings.json` file. Refer to the above section on how to configure it.
2. Run the framework with `python main.py --nb_jobs n` where `n = how many trainings will be done in paralel`.
3. When the code have finished, run `python evaluate_results.py` to choose the best hyperparameters and create the heatmaps with the results.

# How to configure settings.json
- **General**:
    - **training_seed**: Starting training seed for the different seeds will be used.
    - **nb_training_seeds**: How many seeds will be done starting from the training_seed.
    - **device**: Device used in PyTorch to do the computations with torch tensors.
    - **nb_workers**: Number of workers the training will have. If null, this number will be nb_honest + nb_byz.
    - **nb_honest**: Number of honest clients.
    - **nb_byz**: Number of real byzantine clients.
    - **declared_nb_byz**: Number of byzantine clients the server is protecting against.
    - **declared_equal_real**: Boolean to filter configurations that don't have the same declared byzantine clients as the real ones.
    - **size_train_set**: Proportion of the training set that will be used for training; otherwise, it will be used as the validation set.
    - **nb_steps**: Number of iterations that the learning algorithm will perform.
    - **evaluation_delta**: Number of steps between evaluations of the model over the validation set or test set.
    - **evaluate_on_test**: Boolean to configure if test accuracy should be computed.
    - **store_training_accuracy**: Boolean to set if training accuracies will be stored.
    - **store_training_loss**: Boolean to set if training losses will be stored.
    - **store_models**: Boolean to set if the state dict of the PyTorch models should be stored every evaluation delta.
    - **batch_size_validation**: Batch size for the validation and test datasets, but not for the training set.
    - **data_folder**: Where the datasets will be stored.
    - **results_directory**: Where the information we desire to save of the training will be stored.

- **Model**:
    - **name**: Name of the model we want to use in the training. This name must be the name of a class in the **models.py** file.
    - **dataset_name**: Name of the dataset used in training. Must be one of **dataset.py**.
    - **nb_labels**: Number of different targets that the used dataset has.
    - **data_distribution_seed**: Starting data distribution seed for the different seeds will be used. This seed is used to have different data splits among nodes.
    - **nb_data_distribution_seeds**: How many seeds will be done starting from the data_distribution_seed.
    - **data_distribution**:
        - **name**: Name of the data distribution that will be used.
        - **distribution_parameter**: Float with the parameter that will be used by the distribution.
    - **loss**: Pytorch training loss.

- **Aggregator**:
    - **name**: Name of the aggregation used by the server. Must be one of the **aggregators.py**.
    - **parameters**: Dictionary with the parameters required by the aggregation used.

- **PreAggregators**: List of pre-aggregations in order as they should be applied.
    - **name**: Name of the pre-aggregation used by the server. Must be defined in **preaggregators.py**.
    - **parameters**: Dictionary with the parameters required by the pre-aggregations used.

- **Server**:
    - **batch_norm_momentum**: momentum for the federated batch norm. 
    - **learning_rate_decay**: Factor of learning rate decay that will be applied every milestone.
    - **milestones**: List of steps at which the learning rate decay should be applied.

- **Honest Nodes**:
    - **momentum**: Momentum value used in the training algorithm.
    - **batch_size**: Batch size for the SGD algorithm.
    - **learning_rate**: Learning rate value.
    - **weight_decay**: Weight decay value to avoid overfitting.

- **Attack**:
    - **name**: Name of the attack used by the server. Must be defined in **attacks.py**.
    - **parameters**: Dictionary with the parameters required by the attack used.
    - **attack_optimizer**:
        - **name**: Name of the optimizer that will be used to optimize the attack factor.
        - **parameters**: Dictionary with the parameters required by the optimizer.

# How to configure settings.json for several settings
Please note that this library is designed to run multiple settings simultaneously, which is essential for thoroughly exploring aggregations and attacks. To achieve this, you can provide a list of elements in the settings.json file instead of a single element. Below is an example:
`distribution_parameter: 1.0` -> `distribution_parameter: [1.0, 0.5, 0.0]`