import os
import re
import logging
from datasets import load_dataset, load_from_disk, concatenate_datasets, Value

logger = logging.getLogger(__name__)

# All available SuperGLUE tasks
SUPERGLUE_TASKS = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"]



# ---------------------------------------------------------------------------
# Text formatting — one function per task, each producing a single "text"
# field that a causal LM can consume.
# ---------------------------------------------------------------------------

def _format_boolq(ex):
    label = "yes" if ex["label"] == 1 else "no"
    return f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer: {label}"


def _format_cb(ex):
    label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}
    return (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Relation: {label_map[ex['label']]}"
    )


def _format_copa(ex):
    question_str = (
        "What was the cause?" if ex["question"] == "cause"
        else "What happened as a result?"
    )
    answer = ex[f"choice{ex['label'] + 1}"]
    return f"Premise: {ex['premise']}\n{question_str}\nAnswer: {answer}"


def _format_multirc(ex):
    # HF flattens MultiRC: each row is a (paragraph, question, answer, label) tuple
    label_str = "true" if ex["label"] == 1 else "false"
    return (
        f"Paragraph: {ex['paragraph']}\n"
        f"Question: {ex['question']}\n"
        f"Answer candidate: {ex['answer']} — {label_str}"
    )


def _format_record(ex):
    # 'answers' is a list; may be empty for some val rows
    answers = ", ".join(ex.get("answers") or [])
    return (
        f"Passage: {ex['passage']}\n"
        f"Query: {ex['query']}\n"
        f"Answer: {answers}"
    )


def _format_rte(ex):
    label_str = "entailment" if ex["label"] == 0 else "not entailment"
    return (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Relation: {label_str}"
    )


def _format_wic(ex):
    label_str = "true" if ex["label"] == 1 else "false"
    return (
        f"Word: {ex['word']}\n"
        f"Sentence 1: {ex['sentence1']}\n"
        f"Sentence 2: {ex['sentence2']}\n"
        f"Same sense: {label_str}"
    )


def _format_wsc(ex):
    label_str = "true" if ex["label"] else "false"
    return (
        f"Text: {ex['text']}\n"
        f"Does '{ex['span2_text']}' refer to '{ex['span1_text']}'? {label_str}"
    )


_FORMATTERS = {
    "boolq":   _format_boolq,
    "cb":      _format_cb,
    "copa":    _format_copa,
    "multirc": _format_multirc,
    "record":  _format_record,
    "rte":     _format_rte,
    "wic":     _format_wic,
    "wsc":     _format_wsc,
}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _add_text_field(dataset, task):
    """Add a unified 'text' column using the task-specific formatter."""
    formatter = _FORMATTERS[task]
    return dataset.map(lambda ex: {"text": formatter(ex)})


def _tokenize(examples, tokenizer):
    cleaned = [_clean_text(t) for t in examples["text"]]
    return tokenizer(
        cleaned,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

_KEEP_COLUMNS = {"text", "input_ids", "attention_mask"}

def _normalize_features(dataset):
    cols_to_remove = [col for col in dataset.column_names if col not in _KEEP_COLUMNS]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    for col, feature in dataset.features.items():
        if hasattr(feature, "names"):
            dataset = dataset.cast_column(col, Value("int64"))

    return dataset

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_datasets(
    tokenizer,
    batch_size,
    tasks=None,
    reduction_fraction=0.1,
    seed=42,
):
    """
    Load, preprocess, and (optionally) reduce SuperGLUE tasks for use with
    LLaMA-based models. Mirrors the TinyStories loader interface.

    Args:
        tokenizer:            HuggingFace tokenizer.
        batch_size:           Batch size used during dataset.map() preprocessing.
        tasks:                List of SuperGLUE task names to include, or None for
                              all tasks (see SUPERGLUE_TASKS).
        reduction_fraction:   Fraction of each task's data to keep (0 < f <= 1.0).
                              A per-task floor is applied to avoid tasks with
                              too few samples for meaningful evaluation.
        seed:                 Random seed for shuffling / selection.

    Returns:
        train_set, val_set — concatenated across all requested tasks.
    """
    if tasks is None:
        tasks = SUPERGLUE_TASKS

    unknown = set(tasks) - set(SUPERGLUE_TASKS)
    if unknown:
        raise ValueError(f"Unknown SuperGLUE task(s): {unknown}. "
                         f"Valid options: {SUPERGLUE_TASKS}")

    os.makedirs("./data", exist_ok=True)

    # Cache key encodes every factor that affects the final tensors
    task_key = "-".join(sorted(tasks)) if len(tasks) < len(SUPERGLUE_TASKS) else "all"
    cache_train = f"./data/superglue_{task_key}_train_{seed}"
    cache_val   = f"./data/superglue_{task_key}_val_{seed}"

    if os.path.exists(cache_train) and os.path.exists(cache_val):
        train_set = load_from_disk(cache_train)
        val_set   = load_from_disk(cache_val)
        logger.info("SuperGLUE datasets loaded from disk.")
    else:
        train_splits, val_splits = [], []

        for task in tasks:
            logger.info(f"Loading SuperGLUE task: {task}")
            raw_train = load_dataset("super_glue", task, split="train")
            raw_val   = load_dataset("super_glue", task, split="validation")

            raw_train = _add_text_field(raw_train, task)
            raw_val   = _add_text_field(raw_val, task)

            raw_train = _normalize_features(raw_train)
            raw_val = _normalize_features(raw_val)

            raw_train = raw_train.map(
                lambda ex: _tokenize(ex, tokenizer),
                batched=True,
                batch_size=batch_size,
            )
            raw_val = raw_val.map(
                lambda ex: _tokenize(ex, tokenizer),
                batched=True,
                batch_size=batch_size,
            )

            train_splits.append(raw_train)
            val_splits.append(raw_val)
            logger.debug(f"Task '{task}' preprocessed — "
                         f"train: {len(raw_train)}, val: {len(raw_val)}")

        train_set = concatenate_datasets(train_splits)
        val_set   = concatenate_datasets(val_splits)

        train_set.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
        val_set.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

        train_set.save_to_disk(cache_train)
        val_set.save_to_disk(cache_val)
        logger.info("SuperGLUE datasets saved to disk.")

    # Reduce — use a per-task-aware floor: smallest tasks (CB, WSC) have
    # ~250 / ~550 train examples, so a floor of 100 is kept but won't matter
    # much when reduction_fraction is already low.
    if reduction_fraction < 1.0:
        train_size = max(100, int(len(train_set) * reduction_fraction))
        val_size   = max(20,  int(len(val_set)   * reduction_fraction))
        train_set  = train_set.shuffle(seed=seed).select(range(train_size))
        val_set    = val_set.shuffle(seed=seed).select(range(val_size))
        logger.info(
            f"SuperGLUE datasets reduced to {reduction_fraction * 100:.1f}% "
            f"({train_size} train / {val_size} val examples)."
        )

    return train_set, val_set