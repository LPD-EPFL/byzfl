# Import run_benchmark as it is the main function to run the benchmark
from .benchmark import run_benchmark
# Import the functions to evaluate the results of the benchmark
from .evaluate_results import find_best_hyperparameters, heat_map_test_accuracy, aggregated_heat_map_test_accuracy, plot_accuracy_fix_agg_best_setting