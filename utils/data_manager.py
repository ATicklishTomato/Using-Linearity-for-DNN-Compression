from transformers import LlamaTokenizer


class DataManager:
    def __init__(self, dataset_name, batch_size):
        self.dataset_name = dataset_name
        self.batch_size = batch_size

        match dataset_name:
            case "tiny_stories":
                self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to eos token if not already set to prevent errors
                from utils.tinystories import load_datasets
                self.train_set, self.val_set = load_datasets(self.tokenizer, self.batch_size)
            case _:
                raise ValueError(f"Unsupported dataset: {dataset_name}.")