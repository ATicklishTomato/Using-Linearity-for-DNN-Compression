from transformers import LlamaTokenizer


class DataManager:
    def __init__(self, dataset_name, batch_size, reduction_fraction=0.1, seed=42):
        """
        Encapsulating class that manages the loading and preprocessing of datasets for different experiments.
        :param dataset_name: The name of the dataset to load. Supported datasets: "tiny_stories".
        :param batch_size: The batch size to use for loading the dataset.
        :param reduction_fraction: The fraction of the dataset to use to reduce the size for faster experimentation.
        :param seed: The random seed to use for shuffling the dataset when reducing its size.
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        match dataset_name:
            case "tiny_stories":
                self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token if not already set to prevent errors
                from utils.tinystories import load_datasets
                self.train_set, self.val_set = load_datasets(self.tokenizer, self.batch_size, reduction_fraction, seed)
            case "imagenet":
                from utils.imagenet import load_datasets
                self.train_set, self.val_set = load_datasets(reduction_fraction, seed)
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}.")