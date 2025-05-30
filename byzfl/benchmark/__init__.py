# Import run_benchmark as it is the main function to run the benchmark
from .benchmark import run_benchmark
# Import the functions to evaluate the results of the benchmark
from .evaluate_results import test_accuracy_curve, loss_heatmap, test_heatmap, aggregated_test_heatmap