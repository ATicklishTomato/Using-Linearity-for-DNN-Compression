import os

from datasets import load_dataset
import re
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Tokenize and clean the dataset
def preprocess(examples, tokenizer):
    examples["text"] = [clean_text(t) for t in examples["text"]]
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


def load_datasets(tokenizer, batch_size):

    if not os.path.exists("./data"):
        os.makedirs("./data")

    if os.path.exists("./data/tiny_stories_train") and os.path.exists("./data/tiny_stories_val"):
        train_set = load_dataset("roneneldan/TinyStories", split="train").load_from_disk("./data/tiny_stories_train")
        val_set = load_dataset("roneneldan/TinyStories", split="validation").load_from_disk("./data/tiny_stories_val")
        logger.info("Datasets loaded from disk.")
        return train_set, val_set

    train_set = load_dataset("roneneldan/TinyStories", split="train")
    val_set = load_dataset("roneneldan/TinyStories", split="validation")

    train_set = train_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
    val_set = val_set.map((lambda x: preprocess(x, tokenizer)), batched=True, batch_size=batch_size)
    logger.debug("Datasets loaded and preprocessed.")

    # Save to disk for faster loading later
    train_set.save_to_disk("./data/tiny_stories_train")
    val_set.save_to_disk("./data/tiny_stories_val")
    logger.info("Datasets saved to disk.")

    return train_set, val_set