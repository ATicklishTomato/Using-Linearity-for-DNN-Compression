from transformers import LlamaTokenizer
import logging

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, dataset_name, batch_size, model_name=None, reduction_fraction=0.1, seed=42):
        """
        Encapsulating class that manages the loading and preprocessing of datasets for different experiments.
        Args:
            dataset_name (str): Name of the dataset.
            batch_size (int): Batch size for training and evaluation.
            model_name (str, optional): Name of the model to determine the tokenizer to preload. Required for text datasets. Defaults to None.
            reduction_fraction (float, optional): Fraction of the dataset to use for faster experimentation. Defaults to 0.1.
            seed (int, optional): Random seed for reproducibility when reducing the dataset. Defaults to 42.
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        match model_name:
            case "llama-2-7b":
                model_name = "meta-llama/Llama-2-7b-hf"
            case "llama-2-13b":
                model_name = "meta-llama/Llama-2-13b-hf"
            case "llama-3-1b":
                model_name = "meta-llama/Llama-3.2-1B"
            case "llama-3-3b":
                model_name = "meta-llama/Llama-3.2-3B"
            case _:
                if model_name is not None:
                    raise ValueError(f"Unsupported model: {model_name}.")
                else:
                    logger.info("No model name provided.")

        match dataset_name:
            case "tinystories":
                if model_name is None:
                    raise ValueError("Model name must be provided for text datasets to initialize the tokenizer.")

                self.tokenizer = LlamaTokenizer.from_pretrained(model_name)  # Preload tokenizer for text datasets
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token if not already set to prevent errors
                logger.info(f"Tokenizer initialized with pretrained model: {model_name}")

                from utils.tinystories import load_datasets
                self.train_set, self.val_set = load_datasets(self.tokenizer, self.batch_size, reduction_fraction, seed)
            case "imagenet":
                from utils.imagenet import load_datasets
                self.train_set, self.val_set = load_datasets(reduction_fraction, seed)
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}.")


        logger.info(f"DataManager initialized with dataset: {dataset_name}, batch size: {batch_size}, model name: {model_name}, reduction fraction: {reduction_fraction}, seed: {seed}.")