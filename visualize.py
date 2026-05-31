import argparse
import json
import glob
import re
from typing import List

import numpy as np
from itertools import product
import pandas as pd
from matplotlib import pyplot as plt

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
    'superglue': 'SuperGLUE',
}

pretty_linearity_names = {
    'mean_preactivation': 'Mean of preactivations',
    'fraction': 'Fraction of neuron activations',
    'procrustes': 'Procrustes-based linearity score'
}

pretty_benchmark_names = {
    'magnitude_pruning': 'Magnitude pruning',
    'hessian_pruning': 'Hessian pruning',
    'taylor_pruning': 'Taylor pruning',
    'wanda_pruning': 'Wanda pruning',
    'slicegpt': 'SliceGPT pruning',
    'basic_kd': 'Logit-based KD',
    'feature_kd': 'Feature-based KD',
    'born_again_kd': 'Born-again KD'
}


def parse_args():
    """Parses command line arguments for result aggregation and visualization."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--rq', type=str, choices=['rq1', 'rq2', 'benchmark', 'hybridization'], default=['rq1'], nargs='+',
                        help='Which Research Question to aggregate results for')
    parser.add_argument('--threshold', type=str, choices=['float', '25', '50', '75'], default=['75'], nargs='+',
                        help='Threshold to aggregate results for')
    parser.add_argument('--model', type=str, nargs='+',
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-2-13b', 'llama-3-1b', 'llama-3-3b'],
                        default=['resnet18'],
                        help='Which model to aggregate results for')
    parser.add_argument('--dataset', type=str, choices=['imagenet', 'tinystories', 'cifar10', 'superglue'],
                        default=['imagenet'], nargs='+',
                        help='Which dataset to aggregate results for')
    parser.add_argument('--relation_to', type=str, nargs='+',
                        choices=['magnitude_pruning', 'basic_kd', 'hessian_pruning', 'taylor_pruning', 'feature_kd', 'born_again_kd', 'slicegpt', 'wanda_pruning'],
                        default=['magnitude_pruning'], help='Which relation type to aggregate results for')
    parser.add_argument('--linearity', type=str, nargs='+',
                        choices=['mean_preactivation', 'procrustes', 'fraction'],
                        default=['mean_preactivation'],
                        help='Linearity metric to use. `mean_preactivation` refers to the mean of preactivations as defined by Pinson et al. (2024). ' +
                             '`procrustes` refers to the Procrustes similarity-based metric as defined by Razzhigaev et al (2024). ' +
                             '`fraction` refers to the fraction of neurons that is activated by an activation function.')
    parser.add_argument('--approx', action='store_true',
                        help='Whether to aggregate results for linear approximator method instead of layer merging for RQ1. Only applicable for ResNets.')
    return parser.parse_args()

def mean_rq1_results(path):
    """This function reads all *results.json files in the given path, extracts the 'accuracy_loss', 'param_compression_ratio',
    'speedup', and 'gflop_reduction' metrics, and computes the mean for each metric across all files.
    It returns a dictionary containing the mean values for each metric.
    Args:
        path: The directory path where the *results.json files are located. The function will search for these files recursively
                within the directory and its subdirectories.
    Returns:
        A dictionary with the mean values for 'accuracy_loss', 'param_compression_ratio', 'speedup', and 'gflop_reduction'.
                The keys of the dictionary are the metric names, and the values are the corresponding mean values computed from all the *results.json files.
    """
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

def generate_latex_results_table(mean_results, model, dataset, linearity, threshold, path, compression_method):
    """This function takes a dictionary of mean results, and generates a latex table. Gets stored in the results directory
    Args:
        mean_results: A dictionary containing the mean values for 'accuracy_loss', 'param_compression_ratio', 'speedup', and 'gflop_reduction'.
        model: The name of the model (e.g., 'resnet18', 'llama-2-7b').
        dataset: The name of the dataset (e.g., 'imagenet', 'tinystories').
        linearity: The linearity metric used (e.g., 'mean_preactivation', 'fraction', 'procrustes').
        threshold: The threshold used for the linearity metric (e.g., '0', '5', '25', '50', '75').
        path: The directory path where the results are stored. The generated LaTeX table will be saved in this directory as 'results.tex'.
        compression_method: The compression method used (e.g., 'Layer merging', 'Linear approximators', 'Magnitude pruning', etc.)
    """


    if threshold == '0':
        pretty_threshold = "0.0"
    elif threshold == '5':
        pretty_threshold = "0.5"
    else:
        pretty_threshold = threshold + "\\%"
    lines = []
    lines.append("\\begin{table}[p]")
    lines.append("\\begin{center}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append("\\begin{tabular}{|c|c|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Linearity metric} & \\textbf{Threshold} & \\textbf{Model} & \\textbf{Dataset}& \\textbf{Compression method} & \\textbf{Accuracy Loss$\\downarrow$} & \\textbf{Param Compression Ratio$\\downarrow$} & \\textbf{GFLOP Reduction$\\uparrow$} \\\\\\hline")
    lines.append(f"{pretty_linearity_names[linearity]} & {pretty_threshold} & {pretty_model_names[model]} & {pretty_dataset_names[dataset]} & {compression_method} & {mean_results['accuracy_loss']:.4f} & {mean_results['param_compression_ratio']:.2f} & {mean_results['gflop_reduction']:.2f} \\\\\\hline")
    lines.append("\\end{tabular}}")
    lines.append("\\end{center}")
    lines.append("\\end{table}")

    table = "\n".join(lines)

    with open(path + '/results.tex', 'w') as f:
        f.write(table)

def mean_benchmark_results(path):
    """This function reads all wandb_logging_data.json files in the given path, extracts the 'accuracy_loss', 'param_compression_ratio',
    'speedup', and 'gflop_reduction' metrics, and computes the mean for each metric across all files.
    It returns a dictionary containing the mean values for each metric.
    Args:
        path: The directory path where the wandb_logging_data.json files are stored.
    Returns:
        A dictionary with the mean values for 'accuracy_loss', 'param_compression_ratio', 'speedup', and 'gflop_reduction'.
                The keys of the dictionary are the metric names, and the values are the corresponding mean values computed from all the wandb_logging_data.json files.
    """
    metrics = ['accuracy_loss', 'param_compression_ratio', 'speedup', 'gflop_reduction']
    results = {metric: [] for metric in metrics}
    files = glob.glob(path + '/**/wandb_logging_data.json', recursive=True)
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

    mean_results = {metric: sum(values) / len(values) for metric, values in results.items()}
    return mean_results


def avg_rq2_prune_scores(path):
    """This function reads all wandb_logging_data.json files and prune_dict.json files in the given path,
     extracts the 'linearity_scores_mean_preactivation', 'linearity_scores_fraction', 'linearity_scores_procrustes'
     metrics, and pruning ratios, and computes the mean for each metric across all files.
     It returns dictionaries containing the mean values for each metric and pruning ratios.
    Args:
        path: The directory path where the wandb_logging_data.json files are stored.
    Returns:
        A dictionary with the mean values for each metric and pruning ratios.
    """
    mean_preactivations = []
    fraction_scores = []
    procrustes_scores = []
    pruning_ratios = {}
    files = glob.glob(path + '/**/*.json', recursive=True)
    print(files)
    for file in files:
        with open(file, 'r') as f:
            if "wandb_logging_data" in file:
                data = json.load(f)
                # If keys are present in accumulating dict, append values to the value list. Otherwise, create new key with single entry list
                mean_preactivations.append(data['linearity_scores_mean_preactivation'])
                fraction_scores.append(data['linearity_scores_fraction'])
                procrustes_scores.append(data['linearity_scores_procrustes'])
            elif "prune_dict" in file:
                data = json.load(f)
                for key, value in data.items():
                    if key in pruning_ratios.keys():
                        pruning_ratios[key].append(value)
                    else:
                        pruning_ratios[key] = [value]

    avg_mean_preactivation = {}
    avg_mean_fraction = {}
    avg_mean_procrustes = {}
    avg_pruning_ratios = {}
    for key in fraction_scores[0].keys():
        avg_mean_preactivation[key] = np.mean([score[key] for score in mean_preactivations])
        avg_mean_fraction[key] = np.mean([score[key] for score in fraction_scores])
        avg_mean_procrustes[key] = np.mean([score[key] for score in procrustes_scores])
    for key, value in pruning_ratios.items():
        avg_pruning_ratios[key] = np.mean(value)

    return avg_mean_preactivation, avg_mean_fraction, avg_mean_procrustes, avg_pruning_ratios

def avg_rq2_linearity_scores(path):
    """Gets all the averaged linearity metric scores.
    Args:
        path: The directory path where the wandb_logging_data.json files are stored.
    Returns:
        A dictionary with the average linearity metric scores for each metric and pruning ratios.
    """
    mean_preactivations = []
    fraction_scores = []
    procrustes_scores = []
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
            elif "wandb_logging_data" in file:
                data = json.load(f)
                mean_preactivations.append(data['linearity_scores_mean_preactivation'])
                fraction_scores.append(data['linearity_scores_fraction'])
                procrustes_scores.append(data['linearity_scores_procrustes'])

    avg_mean_preactivation = {}
    avg_fraction = {}
    avg_procrustes = {}
    for key in fraction_scores[0].keys():
        avg_mean_preactivation[key] = np.mean([score[key] for score in mean_preactivations])
        avg_fraction[key] = np.mean([score[key] for score in fraction_scores])
        avg_procrustes[key] = np.mean([score[key] for score in procrustes_scores])

    return avg_mean_preactivation, avg_fraction, avg_procrustes, student_layer_names, teacher_layer_names

def scatterplot_linearity_pruning_scores(
    mean_preactivations: List[dict],
    fractions: List[dict],
    procrustes: List[dict],
    pruning_ratios: List[dict],
    dataset_labels: List[str],
    save_dir: str,
) -> None:
    """
    Creates a figure with three scatterplots in a single row:
        1. Mean preactivation vs pruning ratio
        2. Fraction activation vs pruning ratio
        3. Procrustes score vs pruning ratio

    Each dataset is plotted separately and distinguished by color/label.

    Args:
        mean_preactivations:
            List of dictionaries mapping layer names to mean preactivation scores.

        fractions:
            List of dictionaries mapping layer names to fraction activation scores.

        procrustes:
            List of dictionaries mapping layer names to procrustes scores.

        pruning_ratios:
            List of dictionaries mapping layer names to pruning ratios.

        dataset_labels:
            Labels for each dataset.

        save_dir:
            Directory to save the resulting figure.
    """
    fig, axes = plt.subplots(
        len(dataset_labels),
        3,
        figsize=(6 * 3, 5 * len(dataset_labels)),
        squeeze=False,
        sharey="row",
    )

    metric_info = [
        (pretty_linearity_names['mean_preactivation'], mean_preactivations),
        (pretty_linearity_names['fraction'], fractions),
        (pretty_linearity_names['procrustes'], procrustes),
    ]

    for row_idx, dataset_label in enumerate(dataset_labels):

        pruning_dict = pruning_ratios[row_idx]

        for col_idx, (metric_name, metric_dicts) in enumerate(metric_info):
            ax = axes[row_idx, col_idx]

            metric_dict = metric_dicts[row_idx]

            layer_names = list(
                set(metric_dict.keys()).intersection(pruning_dict.keys())
            )

            x_values = [metric_dict[name] for name in layer_names]
            y_values = [pruning_dict[name] for name in layer_names]

            ax.scatter(x_values, y_values, alpha=0.8)

            ax.grid(True)

            # Remove axis labels
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Column labels (top row only)
            if row_idx == 0:
                ax.set_title(metric_name, fontsize=14)

            # Row labels (left column only)
            if col_idx == 0:
                ax.annotate(
                    dataset_label,
                    xy=(-0.175, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=14,
                )

    fig.supylabel("Fraction of Pruned Weights", fontsize=16, x=0.0)

    plt.tight_layout()

    prune_method = save_dir.split("/")[4]
    model_name = save_dir.split("/")[5]

    save_dir = "/".join(save_dir.split("/")[:-2]) + f"/scatterplot_{model_name}_{prune_method}.png"

    plt.savefig(
        save_dir,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def combined_cka_similarity_matrix(matrix, save_dir, teacher_layer_names, student_layer_names,
                                    mean_preactivations, fractions, procrustes_scores):
    """Creates a magma heatmap of the cka similarity matrix using matplotlib.
    Shows row and column indexes to roughly identify layers. Teacher labels are annotated with linearity scores
    Stores the visualization in ./results with the given filename.
    Args:
        matrix: A numpy array of shape (m, n) containing the cka similarity values between the layers.
        save_dir: The directory to save the heatmap. Saved as "cka_similarity_heatmap.png" in the given directory.
        teacher_layer_names: Names of the teacher layers.
        student_layer_names: Names of the student layers.
        mean_preactivations: A dictionary mapping teacher layer names to mean preactivation values.
        fractions: A dictionary mapping teacher layer names to fraction scores.
        procrustes_scores: A dictionary mapping teacher layer names to procrustes scores.
    """

    # Get the matrix row indices for teacher labels that have a score to keep those
    scored_labels = [name for name in teacher_layer_names if
                     name in mean_preactivations.keys() and name in fractions.keys() and name in procrustes_scores.keys()]
    row_indices = sorted([teacher_layer_names.index(name) for name in scored_labels])
    column_indices = sorted([student_layer_names.index(name) for name in set(scored_labels).intersection(student_layer_names)])

    # Drop all but indexed rows
    matrix = matrix[row_indices, :]
    matrix = matrix[:, column_indices]
    if "llama" in save_dir:
        teacher_labels = sorted([teacher_layer_names[i] for i in row_indices], key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1)))
        student_labels = sorted([student_layer_names[i] for i in column_indices], key=lambda s: int(re.search(r"layers\.(\d+)\.", s).group(1)))
    else:
        teacher_labels = sorted([teacher_layer_names[i] for i in row_indices])
        student_labels = sorted([student_layer_names[i] for i in column_indices])

    n_teacher = len(teacher_labels)
    n_student = len(student_labels)

    fig_height = max(8, n_teacher * 0.22)
    fig_width = max(10, n_student * 0.35)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    im = ax.imshow(
        matrix,
        cmap="magma",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        origin="upper"
    )

    # ----------------------------------------
    # Sparse ticks
    # ----------------------------------------

    x_step = max(1, n_student // 20)
    y_step = max(1, n_teacher // 20)

    x_ticks = np.arange(0, n_student, x_step)
    y_ticks = np.arange(0, n_teacher, y_step)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [student_labels[i] for i in x_ticks],
        rotation=90,
        fontsize=10
    )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [teacher_labels[i] for i in y_ticks],
        fontsize=10
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_xlabel("Student model", fontsize=14)
    ax.set_ylabel("Teacher model", fontsize=14)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CKA Similarity", fontsize=12)

    plt.tight_layout()

    kd_method = save_dir.split("/")[4]
    model_name = save_dir.split("/")[5]
    dataset = save_dir.split("/")[6]

    plt.savefig(
        save_dir + f"cka_{model_name}_{kd_method}_{dataset}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    save_teacher_metrics_table(
        save_dir + f"teacher_metrics_{model_name}_{kd_method}_{dataset}.png",
        teacher_labels,
        mean_preactivations,
        fractions,
        procrustes_scores
    )

def save_teacher_metrics_table(
    save_path,
    teacher_layer_names,
    mean_preactivations,
    fractions,
    procrustes_scores
):

    """Saves a table of teacher layer names and their corresponding mean preactivation, fraction, and procrustes scores.
    Args:
        save_path: The path to save the table image.
        teacher_layer_names: List of teacher layer names.
        mean_preactivations: A dictionary mapping teacher layer names to mean preactivation values.
        fractions: A dictionary mapping teacher layer names to fraction scores.
        procrustes_scores: A dictionary mapping teacher layer names to procrustes scores.
    """
    def short_name(name):
        """Shortens the layer name for better display in the table. For LLaMA models, it extracts the layer number.
        For ResNet models, it removes the 'layer' prefix.
        Args:
            name: Name of the layer.
        Returns:
            name: Shortened layer name.
        """
        if "llama" in save_path:
            parts = name.split(".")
            return parts[2] if len(parts) >= 3 else name
        else:
            return name.replace('layer', '')

    rows = []

    for name in teacher_layer_names:

        rows.append({
            "Layer": short_name(name),
            "MP": mean_preactivations.get(name, np.nan),
            "F": fractions.get(name, np.nan),
            "P": procrustes_scores.get(name, np.nan),
        })

    df = pd.DataFrame(rows)

    n_rows = len(df)

    fig_height = max(6, n_rows * 0.28)

    fig, ax = plt.subplots(
        figsize=(5, fig_height)
    )

    ax.axis("off")

    table = ax.table(
        cellText=[
            [
                row["Layer"],
                f"{row['MP']:.2f}" if not np.isnan(row["MP"]) else "—",
                f"{row['F']:.2f}" if not np.isnan(row["F"]) else "—",
                f"{row['P']:.2f}" if not np.isnan(row["P"]) else "—",
            ]
            for _, row in df.iterrows()
        ],
        colLabels=["Layer", "MP", "F", "P"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    table.scale(1, 1.5)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True
    )

    plt.close()

def avg_rq2_matrix_values(path):
    """Computes average CKA matrix for KD experiment.
    Args:
        path: The path to find the multiple matrices
    Returns:
        avg_matrix: The average CKA matrix
    """
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

def rq2_parallel_coordinates_metrics_avg_cka(path, matrix, teacher_layer_names, mean_preactivations, fractions, procrustes_scores):
    """Create parallel coordinates plot highlighting average CKA and linearity metrics for teacher model
    Args:
        path: The path to find the multiple matrices
        matrix: The CKA matrices to average
        teacher_layer_names: List of teacher layer names.
        mean_preactivations: A dictionary mapping teacher layer names to mean preactivation values.
        fractions: A dictionary mapping teacher layer names to fraction scores.
        procrustes_scores: A dictionary mapping teacher layer names to procrustes scores.
    """
    # Average CKA matrix across columns
    average_teacher_cka = np.mean(matrix, axis=1)

    available_teacher_layers_names = set(teacher_layer_names).intersection(fractions.keys())
    # Keep only cka average of available layers
    average_teacher_cka = average_teacher_cka[[teacher_layer_names.index(name) for name in available_teacher_layers_names]]
    teacher_layer_names = [name for name in teacher_layer_names if name in available_teacher_layers_names]

    mp_scores = [mean_preactivations.get(name, np.nan) for name in teacher_layer_names]
    f_scores = [fractions.get(name, np.nan) for name in teacher_layer_names]
    p_scores = [procrustes_scores.get(name, np.nan) for name in teacher_layer_names]

    df = pd.DataFrame({
        "Average CKA": average_teacher_cka,
        "Mean Preactivation": mp_scores,
        "Fraction": f_scores,
        "Procrustes": p_scores,
        "Name": teacher_layer_names,
    })

    metric_cols = [
        "Average CKA",
        "Mean Preactivation",
        "Fraction",
        "Procrustes",
    ]

    # Normalize metrics for cleaner visualization
    df_norm = df.copy()
    for col in metric_cols:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5

    # df_norm = df

    # Find top 3 and bottom 3 by Average CKA
    sorted_idx = np.argsort(average_teacher_cka)

    bottom3 = set(sorted_idx[:3])
    top3 = set(sorted_idx[-3:])

    x = np.arange(len(metric_cols))

    plt.figure(figsize=(10, 6))

    for idx, row in df_norm.iterrows():

        y = row[metric_cols].values

        # Highlight top 3
        if idx in top3:
            plt.plot(
                x,
                y,
                linewidth=3,
                alpha=1.0,
                color="blue",
                label="Highest 3 Average CKA" if idx == list(top3)[0] else None,
                zorder=3,
            )

        # Highlight bottom 3
        elif idx in bottom3:
            plt.plot(
                x,
                y,
                linewidth=3,
                alpha=1.0,
                color="orange",
                label="Lowest 3 Average CKA" if idx == list(bottom3)[0] else None,
                zorder=3,
            )

        # Fade all others
        else:
            plt.plot(
                x,
                y,
                linewidth=1,
                alpha=0.2,
                color="gray",
                zorder=1,
            )

    plt.xticks(x, metric_cols)
    plt.ylabel("Normalized Values")
    plt.grid(alpha=0.3)
    plt.legend()

    kd_method = path.split("/")[4]
    model_name = path.split("/")[5]
    dataset = path.split("/")[6]

    plt.tight_layout()

    plt.savefig(
        f"{path}parallel_coordinates_{model_name}_{kd_method}_{dataset}.png"
    )
    plt.close()



if __name__ == '__main__':
    """Visualize results gathered from experiments easily and quickly. Use CLI to indicate the combinations"""
    args = parse_args()

    options = [args.rq, args.threshold, args.model, args.dataset, args.relation_to, args.linearity]
    combinations = list(product(*options))
    failed_combinations = []

    for rq, threshold, model, dataset, relation_to, linearity in combinations:
        try:
            if threshold == 'float':
                if linearity == 'fraction':
                    threshold = "5"
                elif linearity == 'mean_preactivation':
                    threshold = "0"
                elif linearity == 'procrustes':
                    print("Float for procrustes not a supported threshold.")
                    continue
                else:
                    raise ValueError("Invalid float threshold for unsupported linearity.")
            print(f"Aggregating results for RQ: {rq}, Threshold: {threshold}, Model: {model}, Dataset: {dataset}, Relation: {relation_to}" +
              f", Linearity: {linearity}")
            if rq == 'rq1':
                opt_approx = "approx/" if args.approx else ""
                path = f"./results/rq1/{opt_approx}{linearity}/{threshold}/{model}/{dataset}/"
            elif rq == 'rq2':
                path = f"./results/rq2/all/{relation_to}/{model}/{dataset}/"
            elif rq == 'benchmark':
                path = f"./results/rq2/all/{relation_to}/{model}/{dataset}/"
            elif rq == 'hybridization':
                path = f"./results/rq2/{linearity}/{relation_to }_hybrid/{model}/{dataset}/"
            else:
                raise ValueError("Invalid RQ choice. Must be one of 'rq1', 'rq2', or 'benchmark'.")

            print(path)

            match rq:
                case 'rq1':
                    if 'llama' in model or 'approx' in path:
                        compression_method = "Linear approximators"
                    else:
                        compression_method = "Layer merging"
                    mean_results = mean_rq1_results(path)
                    print("Mean results:", mean_results)
                    generate_latex_results_table(mean_results, model, dataset, linearity, threshold, path, compression_method)
                case 'rq2':
                    if relation_to in ['magnitude_pruning', 'hessian_pruning', 'taylor_pruning', 'wanda_pruning', 'slicegpt']:
                        # Compute average scatterplot
                        if 'llama' in model:
                            avg_mp_ts, avg_f_ts, avg_p_ts, avg_pr_ts = avg_rq2_prune_scores("/".join(path.split("/")[:-2]) + "/tinystories/")
                            avg_mp_sg, avg_f_sg, avg_p_sg, avg_pr_sg = avg_rq2_prune_scores("/".join(path.split("/")[:-2]) + "/superglue/")
                            avg_mean_preactivations = [avg_mp_ts, avg_mp_sg]
                            avg_fraction_scores = [avg_f_ts, avg_f_sg]
                            avg_procrustes_scores = [avg_p_ts, avg_p_sg]
                            avg_pruning_ratios = [avg_pr_ts, avg_pr_sg]
                            dataset_names = [pretty_dataset_names['tinystories'], pretty_dataset_names['superglue']]
                        else:
                            avg_mp_in, avg_f_in, avg_p_in, avg_pr_in = avg_rq2_prune_scores("/".join(path.split("/")[:-2]) + "/imagenet/")
                            avg_mp_ci, avg_f_ci, avg_p_ci, avg_pr_ci = avg_rq2_prune_scores("/".join(path.split("/")[:-2]) + "/cifar10/")
                            avg_mean_preactivations = [avg_mp_in, avg_mp_ci]
                            avg_fraction_scores = [avg_f_in, avg_f_ci]
                            avg_procrustes_scores = [avg_p_in, avg_p_ci]
                            avg_pruning_ratios = [avg_pr_in, avg_pr_ci]
                            dataset_names = [pretty_dataset_names['imagenet'], pretty_dataset_names['cifar10']]

                        scatterplot_linearity_pruning_scores(
                            mean_preactivations=avg_mean_preactivations,
                            fractions=avg_fraction_scores,
                            procrustes=avg_procrustes_scores,
                            pruning_ratios=avg_pruning_ratios,
                            dataset_labels=dataset_names,
                            save_dir=path
                        )
                    elif relation_to in ['basic_kd', 'feature_kd', 'born_again_kd']:
                        avg_matrix = avg_rq2_matrix_values(path)
                        avg_mean_preactivation, avg_fraction_scores, avg_procrustes_scores, student_layer_names, teacher_layer_names = avg_rq2_linearity_scores(path)
                        print("Average mean preactivation example:", list(avg_mean_preactivation.items())[:5])
                        print("Average fraction scores example:", list(avg_fraction_scores.items())[:5])
                        print("Average procrustes scores example:", list(avg_procrustes_scores.items())[:5])
                        combined_cka_similarity_matrix(avg_matrix, path, teacher_layer_names, student_layer_names, avg_mean_preactivation, avg_fraction_scores, avg_procrustes_scores)
                        rq2_parallel_coordinates_metrics_avg_cka(path, avg_matrix, teacher_layer_names, avg_mean_preactivation, avg_fraction_scores, avg_procrustes_scores)
                    else:
                        raise NotImplementedError
                case 'benchmark':
                    mean_bench = mean_benchmark_results(path)
                    print("Benchmark results:", mean_bench)
                    generate_latex_results_table(mean_bench, model, dataset, linearity, threshold, path, pretty_benchmark_names[relation_to])
                case 'hybridization':
                    mean_hybrid = mean_benchmark_results(path)
                    print("Hybridization results:", mean_hybrid)
                    generate_latex_results_table(mean_hybrid, model, dataset, linearity, threshold, path, f"{pretty_benchmark_names[relation_to]} hybrid")
                case _:
                    raise ValueError("Invalid RQ choice. Must be one of 'rq1', 'rq2', 'benchmark', or 'hybridization'.")
        except Exception as e:
            failed_combinations.append((rq, threshold, model, dataset, relation_to, linearity, e))

    print(f"Successful combinations: {len(combinations) - len(failed_combinations)}")
    print(f"Failed combinations: {len(failed_combinations)}")
    for failed_combination in failed_combinations:
        print(f"Failed combination: RQ: {failed_combination[0]}, Threshold: {failed_combination[1]}, Model: {failed_combination[2]}, " +
              f"Dataset: {failed_combination[3]}, Relation: {failed_combination[4]}, Linearity: {failed_combination[5]}, Error: {failed_combination[6]}")