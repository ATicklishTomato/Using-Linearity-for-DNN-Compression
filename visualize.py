import re
import matplotlib.pyplot as plt
import numpy as np
import torch


def cosine_similarity_matrix(linearity_compression_model: torch.nn.Module, kd_model: torch.nn.Module) -> np.ndarray:
    """Computes the cosine similarity matrix for the layers of a model compressed with linearity compression and a model compressed with knowledge distillation.
    Args:
        linearity_compression_model: Linearity compression model with m layers (LLama or ResNet)
        kd_model: Knowledge distillation model with n layers (LLama or ResNet)
    Returns:
        A numpy array of shape (m, n) containing the cosine similarity values between the layers of the two models.
    """

    # Define target layers
    if "resnet" in linearity_compression_model.__class__.__name__:
        target_layer_pattern = r".*conv.*"
    elif "llama" in linearity_compression_model.__class__.__name__:
        target_layer_pattern = r"model\.layers\..*\.self_attn"
    else:
        raise ValueError(f"Unrecognized linearity compression model: {linearity_compression_model}")

    # Count layers per model
    m = len([1 for name, module in linearity_compression_model.named_modules() if re.match(target_layer_pattern, name)])
    n = len([1 for name, module in kd_model.named_modules() if re.match(target_layer_pattern, name)])

    matrix = np.zeros((m, n))

    m_counter = 0
    n_counter = 0
    for lin_name, lin_module in linearity_compression_model.named_modules():
        if re.match(target_layer_pattern, lin_name):
            for kd_name, kd_module in kd_model.named_modules():
                if re.match(target_layer_pattern, kd_name):
                    # Compute cosine similarity between the two layers
                    lin_weights = lin_module.weight.data.view(-1)
                    kd_weights = kd_module.weight.data.view(-1)
                    cos_sim = torch.nn.functional.cosine_similarity(lin_weights, kd_weights, dim=0).item()
                    matrix[m_counter, n_counter] = cos_sim
                    n_counter += 1
            m_counter += 1
            n_counter = 0

    return matrix


def average_cosine_similarity_matrix(matrix: np.ndarray, *args) -> np.ndarray:
    """Computes the average cosine similarity matrix for the layers of a model.
    Args:
        matrix: A numpy array of shape (m, n) containing the cosine similarity values between the layers.
        *args: Any number of matrices also of shape (m, n) containing the cosine similarity values.
    """
    # Verify all arguments are matrices and have the same shape
    shape = matrix.shape
    m = shape[0]
    n = shape[1]
    for mat in args:
        if type(mat) != np.ndarray:
            raise ValueError(f"All arguments must be numpy arrays. Found argument of type {type(mat)}")
        if mat.shape != shape:
            raise ValueError(f"All arguments must have the same shape. Found argument with shape {mat.shape} and expected shape {shape}")

    matrices = np.array([matrix] + list(args))

    # Compute the average of all matrices
    return np.mean(matrices, axis=0)


def visualize_cosine_similarity_matrix(matrix: np.ndarray, filename: str) -> None:
    """Creates a magma heatmap of the cosine similarity matrix using matplotlib.
    Shows row and column indexes to roughly identify layers. Similarity scores are listed in the cells.
    Stores the visualization in ./results with the given filename.
    Args:
        matrix: A numpy array of shape (m, n) containing the cosine similarity values between the layers.
        filename: The filename to save the heatmap. Saved in ./results with the given filename.
    """
    # Strip any file extensions from filename
    filename = filename.split('.')[0]

    plt.imshow(matrix, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.xlabel('Linearity Compression Model Layers')
    plt.ylabel('Knowledge Distillation Model Layers')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"./results/{filename}.png")
    plt.close()


def average_linearity_pruning_scores(scores: dict, *args) -> dict:
    """Computes the average pruning or linearity scores for a number of dictionaries.
    Args:
        scores: A dictionary mapping layer names to scores.
        *args: Any number of dictionaries with the same keys
    Returns:
        A dictionary mapping layer names to average scores.
    """
    # Verify all dicts have the same keys
    original_keys = list(scores.keys())
    for dictionary in args:
        if type(dictionary) != dict:
            raise ValueError(f"All arguments must be dictionaries. Found argument of type {type(dictionary)}")
        if dictionary.keys() != original_keys:
            raise ValueError(f"All arguments must have the same keys. Found argument with keys {dictionary.keys()} and expected keys {original_keys}")

    avg_dict = {}
    all_dicts = [scores] + list(args)
    for key in original_keys:
        avg_dict[key] = np.mean([dictionary[key] for dictionary in all_dicts])

    return avg_dict


def scatterplot_linearity_pruning_scores(linearity_scores: dict, pruning_scores: dict, filename: str) -> None:
    """Creates a scatterplot of the linearity compression scores and pruning scores for each layer.
    Points are labeled with their layer index. X-axis will be linearity score, Y-axis will be pruning score.
    Args:
        linearity_scores: A dictionary mapping layer names to linear scores.
        pruning_scores: A dictionary mapping layer names to pruning scores.
        filename: The filename to save the scatterplot. Saved in ./results with the given filename.
    """
    # Strip any file extensions from filename
    filename = filename.split('.')[0]

    layer_names = list(linearity_scores.keys())
    linearity_values = [linearity_scores[name] for name in layer_names]
    pruning_values = [pruning_scores[name] for name in layer_names]

    plt.figure(figsize=(10, 6))
    plt.scatter(linearity_values, pruning_values)

    for i, name in enumerate(layer_names):
        plt.annotate(name, (linearity_values[i], pruning_values[i]))

    plt.xlabel('Linearity Compression Score')
    plt.ylabel('Pruning Score')
    plt.title('Linearity Compression Scores vs Pruning Scores')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"./results/{filename}.png")
    plt.close()