import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=str, choices=['rq1', 'rq2'], default='rq1',
                        help='Which Research Question to aggregate results for')
    parser.add_argument('--threshold', type=int, default=75,
                        help='Threshold to aggregate results for')
    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-2-13b', 'llama-3-1b', 'llama-3-3b'],
                        default='resnet18',
                        help='Which model to aggregate results for')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'tinystories'], default='imagenet',
                        help='Which dataset to aggregate results for')
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
    table += f"{args.model} & {args.dataset} & {args.threshold} & {mean_results['accuracy_loss']:.4f} & {mean_results['param_compression_ratio']:.4f} & {mean_results['speedup']:.4f} & {mean_results['gflop_reduction']:.4f} \\\\\n"
    table += "\\end{tabular}\n"
    table += "\\end{center}\n"
    table += "\\end{table}\n"

    with open(path + '/results.tex', 'w') as f:
        f.write(table)


if __name__ == '__main__':
    args = parse_args()
    print(f"Aggregating results for RQ: {args.rq}, Threshold: {args.threshold}, Model: {args.model}, Dataset: {args.dataset}")

    path = f"./results/{args.rq}/{args.threshold}/{args.model[:-2]}/{args.dataset}/"

    match (args.rq):
        case 'rq1':
            mean_results = mean_rq1_results(path)
            print("Mean results:", mean_results)
            generate_latex_results_table(mean_results, args, path)
        case 'rq2':
            print("RQ2 results aggregation not implemented yet.")

