import torch
import logging
from metrics.mean_preactivation import mean_preactivations

logger = logging.getLogger(__name__)

class LinearityMetric:

    def __init__(self, metric_name: str, model_name: str, data_handler, threshold, max_batches=None, device='cuda', save=False):
        """Encapsulating class that manages the application of the correct metric implementation for a model.

        Args:
            metric_name (str): The name of the linearity metric to compute. Supported metrics: "mean_preactivation", "procrustes", "fraction".
            model_name (str): The name of the model to compute the metric on. Supported models: "Llama-2-7b", "Llama-2-13b", "Resnet-18", "Resnet-34", "Resnet-50".
            data_handler: An instance of DataManager that provides access to the dataset and tokenizer (if applicable).
            threshold: The threshold to use for determining what is(n't) linear. To take a percentile, enter a percentage, e.g. `75%` to consider anything smaller the 75th percentile as non-linear. To take a hard threshold, enter a floating point value, e.g. `-0.01`. Default is 75th percentile.
            max_batches: Maximum number of batches to process during metric computation. If None, process all batches.
            device: Device to run the computations on (e.g., "cpu", "cuda").
            save: Whether to save the computed metric values to disk for faster loading in future runs. If True, the metric values will be saved to a file named `./results/{metric_name}_{model_name}.pt`. If such a file exists, the metric values will be loaded from the file instead of recomputing them.
        """

        self.metric_name = metric_name
        self.model_name = model_name
        self.data_handler = data_handler
        self.thresholder = self.threshold_fn(threshold)
        self.max_batches = max_batches if max_batches is not None else len(data_handler.val_set)
        self.device = device
        self.save = save

        match (model_name, metric_name):
            case (_, "mean_preactivation"):
                self.metric_fn = lambda model: mean_preactivations(model, self.data_handler,
                                                                         max_batches=self.max_batches,
                                                                         device=self.device, save=self.save)
            case ("llama-2-7b" | "llama-2-13b" | "llama-3-1b" | "llama-3-3b", "procrustes"):
                raise NotImplementedError("Procrustes metric not implemented for Llama yet.")
            case ("llama-2-7b" | "llama-2-13b" | "llama-3-1b" | "llama-3-3b", "fraction"):
                raise NotImplementedError("Fraction metric not implemented for Llama yet.")
            # case ("resnet18" | "resnet34" | "resnet50", "mean_preactivation"):
            #     self.metric_fn = lambda model: mean_preactivations_resnet(model, self.data_handler.val_set,
            #                                                               batch_size=self.data_handler.batch_size,
            #                                                               device=self.device)
            case("resnet18" | "resnet34" | "resnet50", "procrustes"):
                raise NotImplementedError("Procrustes metric not implemented for Resnet yet.")
            case("resnet18" | "resnet34" | "resnet50", "fraction"):
                raise NotImplementedError("Fraction metric not implemented for Resnet yet.")
            case _:
                raise ValueError(f"Unsupported model and metric combination: {model_name} and {metric_name}.")

        logger.info(f"LinearityMetric initialized with model: {model_name}, metric: {metric_name}, threshold: {threshold}, max_batches: {max_batches}, device: {device}, save: {save}.")


    def threshold_fn(self, threshold):
        """Encapsulating a threshold function that either splits based on a percentile or float.
        Args:
            threshold (str): A string that is one of the following: None, a percentage, e.g. `75%`, or a float.
        Returns:
            A function that takes a dictionary of layer names and linearity scores, and splits it into two dictionaries. The first dictionary contains the layers that are considered linear (i.e., those with scores above the threshold), and the second dictionary contains the layers that are considered non-linear (i.e., those with scores below the threshold).
            """
        logger.info(f"Threshold: {threshold}")
        if threshold is None:
            logger.debug(f"Threshold is None, defaulting to 75%")
            thresholder = lambda dictionary: (
                {k: v for k, v in dictionary.items() if v >= torch.quantile(torch.tensor(list(dictionary.values())), 0.75)},
                {k: v for k, v in dictionary.items() if v < torch.quantile(torch.tensor(list(dictionary.values())), 0.75)}
            )
        elif isinstance(threshold, str) and threshold.endswith('%'):
            percentage = float(threshold[:-1]) / 100.0
            logger.debug(f"Threshold is percentage: {percentage}")
            thresholder = lambda dictionary: (
                {k: v for k, v in dictionary.items() if v >= torch.quantile(torch.tensor(list(dictionary.values())), percentage)},
                {k: v for k, v in dictionary.items() if v < torch.quantile(torch.tensor(list(dictionary.values())), percentage)}
            )
        else:
            try:
                float_threshold = float(threshold)
                logger.debug(f"Threshold is float: {float_threshold}")
                thresholder = lambda dictionary: (
                    {k: v for k, v in dictionary.items() if v >= float_threshold},
                    {k: v for k, v in dictionary.items() if v < float_threshold}
                )
            except ValueError:
                raise ValueError(f"Invalid threshold value: {threshold}. Must be None, a percentage string (e.g., '75%'), or a float.")

        return thresholder

