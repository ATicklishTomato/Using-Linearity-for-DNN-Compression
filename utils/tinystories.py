import os

from datasets import load_dataset, Dataset
import re
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    """Cleans the input text by removing unwanted characters and extra spaces.
    Args:
        text: Text to be cleaned.
    Returns:
        clean_text: Cleaned text.
    """
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Tokenize and clean the dataset
def preprocess(examples, tokenizer):
    """Preprocesses the given examples using the given tokenizer.
    Args:
        examples: List of examples to be preprocessed.
        tokenizer: Tokenizer object for tokenizing examples.
    Returns:
        tokenized_examples: Tokenized and cleaned examples.
    """
    examples["text"] = [clean_text(t) for t in examples["text"]]
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


def load_datasets(tokenizer, batch_size, reduction_fraction=0.1, seed=42):
    """Loads and preprocesses the TinyStories dataset, with options for caching and reducing dataset size for faster experimentation.
    Args:
        tokenizer: Tokenizer object for tokenizing TinyStories dataset.
        batch_size: Number of samples per batch.
        reduction_fraction: Fraction of the dataset to use for training and validation (default: 0.1 for faster experimentation).
        seed: Random seed (default: 42).
    Returns:
        train_set: Preprocessed training dataset.
        val_set: Preprocessed validation dataset.
    """

    if not os.path.exists("./data"):
        os.makedirs("./data", exist_ok=True)

    if (os.path.exists(f"./data/tiny_stories_train_{seed}")
            and os.path.exists(f"./data/tiny_stories_val_{seed}")):
        train_set = Dataset.load_from_disk(f"./data/tiny_stories_train_{seed}")
        val_set = Dataset.load_from_disk(f"./data/tiny_stories_val_{seed}")
        logger.info("Datasets loaded from disk.")
    else:
        train_set = load_dataset("roneneldan/TinyStories", split="train")
        val_set = load_dataset("roneneldan/TinyStories", split="validation")

        train_set = train_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
        val_set = val_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
        logger.debug("Datasets loaded and preprocessed.")

        # Save to disk for faster loading later
        train_set.save_to_disk(f"./data/tiny_stories_train_{seed}")
        val_set.save_to_disk(f"./data/tiny_stories_val_{seed}")
        logger.info("Datasets saved to disk.")

    # Reduce dataset size for faster experimentation
    if reduction_fraction < 1.0:
        train_size = max(100, int(len(train_set) * reduction_fraction))
        val_size = max(20, int(len(val_set) * reduction_fraction))
        train_set = train_set.shuffle(seed=seed).select(range(train_size))
        val_set = val_set.shuffle(seed=seed).select(range(val_size))
        logger.info(f"Datasets reduced to {reduction_fraction*100}% of original size for faster experimentation.")

    return train_set, val_set