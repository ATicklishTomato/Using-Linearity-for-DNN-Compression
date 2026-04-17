import argparse

import numpy as np

from experiments.relation import scatterplot_linearity_pruning_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=str, choices=['rq1', 'rq2', 'benchmark'], default='rq1',
                        help='Which Research Question to aggregate results for')
    parser.add_argument('--threshold', type=int, default=75,
                        help='Threshold to aggregate results for')
    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-2-13b', 'llama-3-1b', 'llama-3-3b'],
                        default='resnet18',
                        help='Which model to aggregate results for')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'tinystories'], default='imagenet',
                        help='Which dataset to aggregate results for')
    parser.add_argument('--relation_to', type=str,
                        choices=['magnitude_pruning', 'basic_kd'], default='magnitude_pruning',
                        help='Which relation type to aggregate results for')
    return parser.parse_args()

def mean_rq1_results(path):
    # Read all *results.json files for 'accuracy_loss', 'param_compression_ratio', 'speedup', and 'tflop_reduction' and compute the mean for each metric
    import json
    import glob

    metrics = ['accuracy_loss', 'param_compression_ratio', 'speedup', 'gflop_reduction']
    results = {metric: [] for metric in metrics}
    files = glob.glob(path + '/**/*.json', recursive=True)
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            results['accuracy_loss'].append(data['accuracy_loss'])
            results['param_compression_ratio'].append(data['param_compression_ratio'])
            results['speedup'].append(data['speedup'])
            if data['tflop_reduction'] is not None:
                # Some data was labeled wrongly, this deals with that
                results['gflop_reduction'].append(data['tflop_reduction'])
            else:
                results['gflop_reduction'].append(data['gflop_reduction'])

    mean_results = {metric: sum(values)/len(values) for metric, values in results.items()}
    return mean_results

def generate_latex_results_table(mean_results, args, path):
    """This function takes a dictionary of mean results, and generates a latex table. Gets stored in the results directory"""

    table = "\\begin{table}[H]\n"
    table += "\\caption{Results for " + args.model + " averaged over 5 runs}\n"
    table += "\\label{tab:" + args.model + "_results}\n"
    table += "\\begin{center}\n"
    table += "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    table += "\\textbf{Model} & \\textbf{Dataset} & \\textbf{Threshold} & \\textbf{Accuracy Loss$\\downarrow$} & \\textbf{Param Compression Ratio$\\uparrow$} & \\textbf{Speedup$\\uparrow$} & \\textbf{GFLOP Reduction$\\uparrow$} \\\\\n\\hline\n"
    table += f"{args.model} & {args.dataset} & {args.threshold} & {mean_results['accuracy_loss']:.4f} & {mean_results['param_compression_ratio']:.4f} & {mean_results['speedup']:.4f} & {mean_results['gflop_reduction']} \\\\\n"
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    with open(path + '/results.tex', 'w') as f:
        f.write(table)

def mean_benchmark_results(path, base_metrics_path="./results/rq1/75/resnet/imagenet/**/*.json"):
    import json
    import glob
    comp_metrics = ['comp_acc', 'comp_params', 'comp_infer']#, 'comp_gflops']
    base_metrics = ['original_accuracy', 'original_param_count', 'original_inference_time']#, 'original_gflops']

    # Get the base metrics from all json files in base metrics path and average them out
    base_metrics_files = glob.glob(base_metrics_path, recursive=True)
    base_results = {metric: [] for metric in base_metrics}
    for file in base_metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            base_results['original_accuracy'].append(data['original_accuracy'])
            base_results['original_param_count'].append(data['original_param_count'])
            base_results['original_inference_time'].append(data['original_inference_time'])
            # if data['original_gflops'] is not None:
            #     # Some data was labeled wrongly, this deals with that
            #     base_results['original_gflops'].append(data['original_gflops'])
            # else:
            #     base_results['original_gflops'].append(data['original_tflops'])

    base_results = {metric: sum(values)/len(values) for metric, values in base_results.items()}

    # Get the compression method results and average them out
    comp_metrics_files = glob.glob(path + "*.json")
    comp_results = {metric: [] for metric in comp_metrics}
    print(comp_metrics_files)
    for file in comp_metrics_files:
        with open(file, 'r') as f:
            data = json.load(f)
            comp_results['comp_acc'] = data['comp_acc']
            comp_results['comp_params'] = data['comp_params']
            comp_results['comp_infer'] = data['comp_infer']
            # comp_results['comp_gflops'] = data['comp_gflops']

    comp_results = {metric: sum(values)/len(values) for metric, values in comp_results.items()}

    # Compute accuracy_loss, param_compression_ratio, speedup, and gflop_reduction
    accuracy_loss = base_results['original_accuracy'] - comp_results['comp_acc']
    param_compression_ratio = base_results['original_param_count'] / comp_results['comp_params']
    speedup = base_results['original_inference_time'] / comp_results['comp_infer']
    # gflop_reduction = base_results['original_gflops'] / comp_results['comp_gflops']

    return {
        'accuracy_loss': accuracy_loss,
        'param_compression_ratio': param_compression_ratio,
        'speedup': speedup,
        'gflop_reduction': "Unknown"#gflop_reduction,
    }


def avg_rq2_scores(path):
    import json
    import glob

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


if __name__ == '__main__':
    args = parse_args()
    print(f"Aggregating results for RQ: {args.rq}, Threshold: {args.threshold}, Model: {args.model}, Dataset: {args.dataset}, Relation: {args.relation_to}")
    base_model_name = "resnet" if "resnet" in args.model else "llama"
    if args.rq == 'rq1':
        path = f"./results/{args.rq}/{args.threshold}/{base_model_name}/{args.dataset}/"
    elif args.rq == 'rq2':
        path = f"./results/{args.rq}/{args.relation_to}/{base_model_name}/{args.dataset}/"
    elif args.rq == 'benchmark':
        path = f"./results/rq2/{args.relation_to}/{base_model_name}/{args.dataset}/"
    else:
        raise ValueError("Invalid RQ choice. Must be one of 'rq1', 'rq2', or 'benchmark'.")

    print(path)

    match (args.rq):
        case 'rq1':
            mean_results = mean_rq1_results(path)
            print("Mean results:", mean_results)
            generate_latex_results_table(mean_results, args, path)
        case 'rq2':
            if args.relation_to == 'magnitude_pruning':
                # Compute average scatterplot
                avg_linearity_scores, avg_pruning_ratios = avg_rq2_scores(path)
                print("Average Linearity Scores:", avg_linearity_scores)
                print("Average Pruning Ratios:", avg_pruning_ratios)
                scatterplot_linearity_pruning_scores(avg_linearity_scores, avg_pruning_ratios, path)
            else:
                raise NotImplementedError
        case 'benchmark':
            mean_bench = mean_benchmark_results(path)
            print("Benchmark results:", mean_bench)
            generate_latex_results_table(mean_bench, args, path)

