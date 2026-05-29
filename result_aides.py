import glob
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_layer_metrics(layer_labels, mean_preactivations, fractions, procrustes,
                       path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

    metrics = [mean_preactivations, fractions, procrustes]
    metric_names = ['Mean of preactivations', 'Fraction of neuron activations', 'Procrustes-based linearity score']

    # Build tick positions
    x_positions = list(range(len(layer_labels)))
    tick_positions = x_positions[::1]
    tick_labels = [layer_labels[i] for i in range(0, len(layer_labels), 1)]

    for ax, metric, name in zip(axes, metrics, metric_names):
        ax.plot(layer_labels, metric, marker='o', markersize=3, linewidth=1)

        # Axis labeling
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title(name, fontsize=11)

        # Apply controlled ticks
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Light grid for interpretability
        ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def make_analysis_graphs(model):
    fractions_path = f"./results/rq1/*/*/{model}/tinystories/*/activation_fractions.json"
    preactivations_path = f"./results/rq1/*/*/{model}/tinystories/*/mean_preactivations.json"
    procrustes_path = f"./results/rq1/*/*/{model}/tinystories/*/procrustes_scores.json"

    fraction_files = glob.glob(fractions_path)
    preactivations_files = glob.glob(preactivations_path)
    procrustes_files = glob.glob(procrustes_path)

    labels = list(json.load(open(fraction_files[0])).keys())
    labels = [l.replace("model.", '') for l in labels]
    fraction_scores = np.mean([list(json.load(open(f)).values()) for f in fraction_files], axis=0)
    preactivations_scores = np.mean([list(json.load(open(f)).values()) for f in preactivations_files], axis=0)
    procrustes_scores = np.mean([list(json.load(open(f)).values()) for f in procrustes_files], axis=0)

    print(labels)
    print(fraction_scores)
    print(preactivations_scores)
    print(procrustes_scores)
    print(len(labels), len(fraction_scores), len(preactivations_scores), len(procrustes_scores))

    plot_layer_metrics(labels, preactivations_scores, fraction_scores, procrustes_scores, f"./{model}_tinystories_metrics_comparison.png")

def make_latex_table(model, approx=False):
    approx = "approx/" if approx else ""
    path = f'./results/rq1/{approx}*/*/{model}/*/results.tex'

    tables = glob.glob(path)
    start = ""
    rows = ""
    finish = ""
    for table in tables:
        text = open(table, 'r').read()

        # Find the line that doesnt start with a backslash

        row = [line for line in text.splitlines() if not line.startswith('\\') and line.strip() != ''][0]

        if start == "":
            start = text.split(row)[0]
        if finish == "":
            finish = text.split(row)[-1]

        rows += row + "\n"

    with open(f"./results/rq1/{model}_{approx[:-1]}_results.tex", 'w') as f:
        f.write(start + rows + finish)

def get_average_original_results(model, dataset):
    path = f'./results/rq1/*/*/{model}/{dataset}/*/*results.json'
    results = glob.glob(path)
    accuracies = []
    comp_ratios = []
    for result in results:
        data = json.load(open(result))
        accuracies.append(data["original_accuracy"])
        comp_ratios.append(data["original_param_count"])

    print(f"Model: {model}, Dataset: {dataset}, Average Accuracy: {np.mean(accuracies):.4f}, Average Comp Ratio: {np.mean(comp_ratios):.4f}")

if __name__ == '__main__':

    for model in ['resnet18', 'resnet34', 'resnet50', 'llama-3-1b']:
        datasets = ['tinystories', 'superglue'] if model == 'llama-3-1b' else ['cifar10', 'imagenet']
        for dataset in datasets:
            get_average_original_results(model, dataset)
