import argparse
import json
import glob
import numpy as np
import utils.util_functions as utils
from itertools import product
from experiments.relation import scatterplot_linearity_pruning_scores, visualize_cka_similarity_matrix

pretty_model_names = {
    'resnet18': 'ResNet-18',
    'resnet34': 'ResNet-34',
    'resnet50': 'ResNet-50',
    'llama-2-7b': 'Llama-2-7B',
    'llama-2-13b': 'Llama-2-13B',
    'llama-3-1b': 'Llama-3.2-1B',
    'llama-3-3b': 'Llama-3.2-1B',
}

pretty_dataset_names = {
    'imagenet': 'ImageNet',
    'tinystories': 'TinyStories',
    'cifar10': 'CIFAR-10',
}

pretty_linearity_names = {
    'mean_preactivation': 'mean of preactivations',
    'fraction': 'fraction of neuron activations',
    'procrustes': 'Procrustes-based linearity score'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=str, choices=['rq1', 'rq2', 'benchmark'], default=['rq1'], nargs='+',
                        help='Which Research Question to aggregate results for')
    parser.add_argument('--threshold', type=int, default=[75], nargs='+',
                        help='Threshold to aggregate results for')
    parser.add_argument('--model', type=str, nargs='+',
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-2-13b', 'llama-3-1b', 'llama-3-3b'],
                        default=['resnet18'],
                        help='Which model to aggregate results for')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'tinystories', 'cifar10'],
                        default=['imagenet'], nargs='+',
                        help='Which dataset to aggregate results for')
    parser.add_argument('--relation_to', type=str, nargs='+',
                        choices=['magnitude_pruning', 'basic_kd'], default=['magnitude_pruning'],
                        help='Which relation type to aggregate results for')
    parser.add_argument('--linearity', type=str, nargs='+',
                        choices=['mean_preactivation', 'procrustes', 'fraction'],
                        default=['mean_preactivation'],
                        help='Linearity metric to use. `mean_preactivation` refers to the mean of preactivations as defined by Pinson et al. (2024). ' +
                             '`procrustes` refers to the Procrustes similarity-based metric as defined by Razzhigaev et al (2024). ' +
                             '`fraction` refers to the fraction of neurons that is activated by an activation function.')
    return parser.parse_args()

def mean_rq1_results(path):
    # Read all *results.json files for 'accuracy_loss', 'param_compression_ratio', 'speedup', and 'tflop_reduction' and compute the mean for each metric
    metrics = ['accuracy_loss', 'param_compression_ratio', 'speedup', 'gflop_reduction']
    results = {metric: [] for metric in metrics}
    files = glob.glob(path + '/**/*results.json', recursive=True)
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            results['accuracy_loss'].append(data['accuracy_loss'])
            results['param_compression_ratio'].append(data['param_compression_ratio'])
            results['speedup'].append(data['speedup'])
            if 'tflop_reduction' in data.keys():
                # Some data was labeled wrongly, this deals with that
                results['gflop_reduction'].append(data['tflop_reduction'])
            else:
                results['gflop_reduction'].append(data['gflop_reduction'])

    mean_results = {metric: sum(values)/len(values) for metric, values in results.items()}
    return mean_results

def generate_latex_results_table(mean_results, model, dataset, linearity, threshold, path):
    """This function takes a dictionary of mean results, and generates a latex table. Gets stored in the results directory"""

    caption = (
        f"Results for {pretty_model_names[model]} trained on {pretty_dataset_names[dataset]} " +
        f"with the {pretty_linearity_names[linearity]} linearity metric averaged over 5 runs"
    )

    label = f"tab:{model}_{dataset}_{linearity}_results"

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{center}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Compression method} & "
        "\\textbf{Target} & "
        "\\textbf{Accuracy Loss$\\downarrow$} & "
        "\\textbf{Param Compression Ratio$\\downarrow$} & "
        "\\textbf{Speedup$\\uparrow$} & "
        "\\textbf{GFLOP Reduction$\\uparrow$} \\\\\\hline"
    )
    lines.append(f"Ours & {threshold} & {mean_results['accuracy_loss']*100:.4f}\\% & {mean_results['param_compression_ratio']:.4f} & {mean_results['speedup']:.4f} & {mean_results['gflop_reduction']:.4f} \\\\")
    lines.append("\\end{tabular}}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")

    table = "\n".join(lines)

    with open(path + '/results.tex', 'w') as f:
        f.write(table)

def mean_benchmark_results(path, base_metrics_path="./results/rq1/mean_preactivation/75/resnet18/imagenet/**/*results.json"):
    comp_metrics = ['comp_acc', 'comp_params', 'comp_infer', 'comp_gflops']
    base_metrics = ['original_accuracy', 'original_param_count', 'original_inference_time', 'original_gflops']

    # Get the base metrics from all json files in base metrics path and average them out
    base_metrics_files = glob.glob(base_metrics_path, recursive=True)
    base_results = {metric: [] for metric in base_metrics}
    for file in base_metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            base_results['original_accuracy'].append(data['original_accuracy'])
            base_results['original_param_count'].append(data['original_param_count'])
            base_results['original_inference_time'].append(data['original_inference_time'])
            if 'original_tflops' in data.keys():
                # Some data was labeled wrongly, this deals with that
                base_results['original_gflops'].append(data['original_tflops'])
            else:
                base_results['original_gflops'].append(data['original_gflops'])

    base_results = {metric: sum(values)/len(values) for metric, values in base_results.items()}

    # Get the compression method results and average them out
    comp_metrics_files = glob.glob(path + "*.json")
    comp_results = {metric: [] for metric in comp_metrics}
    for file in comp_metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            comp_results['comp_acc'] = data['comp_acc']
            comp_results['comp_params'] = data['comp_params']
            comp_results['comp_infer'] = data['comp_infer']
            comp_results['comp_gflops'] = data['comp_gflops']

    comp_results = {metric: sum(values)/len(values) for metric, values in comp_results.items()}

    # Compute accuracy_loss, param_compression_ratio, speedup, and gflop_reduction
    accuracy_loss = utils.accuracy_loss(base_results['original_accuracy'], comp_results['comp_acc'])
    param_compression_ratio = utils.compression_ratio(base_results['original_param_count'], comp_results['comp_params'])
    speedup = utils.speedup(base_results['original_inference_time'], comp_results['comp_infer'])
    gflop_reduction = utils.gflop_reduction(base_results['original_gflops'], comp_results['comp_gflops'])

    return {
        'accuracy_loss': accuracy_loss,
        'param_compression_ratio': param_compression_ratio,
        'speedup': speedup,
        'gflop_reduction': gflop_reduction,
    }


def avg_rq2_prune_scores(path):
    linearity_scores = {}
    pruning_ratios = {}
    files = glob.glob(path + '/**/*.json', recursive=True)
    print(files)
    for file in files:
        with open(file, 'r') as f:
            if "linearity_scores" in file:
                data = json.load(f)
                # If keys are present in accumulating dict, append values to the value list. Otherwise, create new key with single entry list
                for key, value in data.items():
                    if key in linearity_scores.keys():
                        linearity_scores[key].append(value)
                    else:
                        linearity_scores[key] = [value]
            elif "prune_dict" in file:
                data = json.load(f)
                for key, value in data.items():
                    if key in pruning_ratios.keys():
                        pruning_ratios[key].append(value)
                    else:
                        pruning_ratios[key] = [value]

    avg_linearity_scores = {}
    avg_pruning_ratios = {}
    for key, value in linearity_scores.items():
        avg_linearity_scores[key] = np.mean(value)
    for key, value in pruning_ratios.items():
        avg_pruning_ratios[key] = np.mean(value)

    return avg_linearity_scores, avg_pruning_ratios

def avg_rq2_linearity_scores(path):
    linearity_scores = {}
    student_layer_names = []
    teacher_layer_names = []
    files = glob.glob(path + '/**/*.json', recursive=True)
    print(files)
    for file in files:
        with open(file, 'r') as f:
            if "student_layer_names" in file and len(student_layer_names) <= 0:
                student_layer_names = list(json.load(f))
            elif "teacher_layer_names" in file and len(teacher_layer_names) <= 0:
                teacher_layer_names = list(json.load(f))
            elif "linearity_scores" in file:
                data = json.load(f)
                # If keys are present in accumulating dict, append values to the value list. Otherwise, create new key with single entry list
                for key, value in data.items():
                    if key in linearity_scores.keys():
                        linearity_scores[key].append(value)
                    else:
                        linearity_scores[key] = [value]

    avg_linearity_scores = {}
    for key, value in linearity_scores.items():
        avg_linearity_scores[key] = np.mean(value)

    return avg_linearity_scores, student_layer_names, teacher_layer_names

def avg_rq2_matrix_values(path):
    files = glob.glob(path + '/**/cka_similarity_matrix.npy', recursive=True)
    print(files)
    avg_matrix = None
    for file in files:
        matrix = np.load(file)
        if avg_matrix is None:
            avg_matrix = matrix
        else:
            avg_matrix += matrix
    avg_matrix /= len(files)
    return avg_matrix

if __name__ == '__main__':
    args = parse_args()

    options = [args.rq, args.threshold, args.model, args.dataset, args.relation_to, args.linearity]
    combinations = list(product(*options))
    failed_combinations = []

    for rq, threshold, model, dataset, relation_to, linearity in combinations:
        try:
            print(f"Aggregating results for RQ: {rq}, Threshold: {threshold}, Model: {model}, Dataset: {dataset}, Relation: {relation_to}" +
              f", Linearity: {linearity}")
            if rq == 'rq1':
                path = f"./results/{rq}/{linearity}/{threshold}/{model}/{dataset}/"
            elif rq == 'rq2':
                path = f"./results/{rq}/{linearity}/{relation_to}/{model}/{dataset}/"
            elif rq == 'benchmark':
                path = f"./results/rq2/{linearity}/{relation_to}/{model}/{dataset}/"
            else:
                raise ValueError("Invalid RQ choice. Must be one of 'rq1', 'rq2', or 'benchmark'.")

            print(path)

            match rq:
                case 'rq1':
                    mean_results = mean_rq1_results(path)
                    print("Mean results:", mean_results)
                    generate_latex_results_table(mean_results, model, dataset, linearity, threshold, path)
                case 'rq2':
                    if relation_to == 'magnitude_pruning':
                        # Compute average scatterplot
                        avg_linearity_scores, avg_pruning_ratios = avg_rq2_prune_scores(path)
                        print("Average Linearity Scores example:", list(avg_linearity_scores.items())[:5])
                        print("Average Pruning Ratios example:", list(avg_pruning_ratios.items())[:5])
                        scatterplot_linearity_pruning_scores(avg_linearity_scores, avg_pruning_ratios, path)
                    elif relation_to == 'basic_kd':
                        avg_matrix = avg_rq2_matrix_values(path)
                        avg_linearity_scores, student_layer_names, teacher_layer_names = avg_rq2_linearity_scores(path)
                        print("Average Linearity Scores example:", list(avg_linearity_scores.items())[:5])
                        visualize_cka_similarity_matrix(avg_matrix, path, teacher_layer_names, student_layer_names,
                                                        avg_linearity_scores)
                    else:
                        raise NotImplementedError
                case 'benchmark':
                    base_performance_path = f"./results/rq1/{linearity}/{threshold}/{model}/{dataset}/**/*results.json"
                    mean_bench = mean_benchmark_results(path, base_metrics_path=base_performance_path)
                    print("Benchmark results:", mean_bench)
                    generate_latex_results_table(mean_bench, model, dataset, linearity, threshold, path)
        except Exception as e:
            failed_combinations.append((rq, threshold, model, dataset, relation_to, linearity, e))

    print(f"Successful combinations: {len(combinations) - len(failed_combinations)}")
    print(f"Failed combinations: {len(failed_combinations)}")
    for failed_combination in failed_combinations:
        print(f"Failed combination: RQ: {failed_combination[0]}, Threshold: {failed_combination[1]}, Model: {failed_combination[2]}, " +
              f"Dataset: {failed_combination[3]}, Relation: {failed_combination[4]}, Linearity: {failed_combination[5]}, Error: {failed_combination[6]}")
