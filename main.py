import os
import random
from argparse import ArgumentParser
import logging
from datetime import datetime
import numpy as np
import torch
import wandb

# Change wandb logging directory so IDEs don't confuse log directory for importable package
os.environ['WANDB_DIR'] = './wandb_logs'

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(description='Execute experiments on inherent linearity in ResNets and Llamas.')
    parser.add_argument('-m', '--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama7b', 'llama13b'],
                        default='resnet18',
                        help='Model architecture to use for the experiment.')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['imagenet', 'tinystories'],
                        default='imagenet',
                        help='Dataset to use for training and evaluation.')
    parser.add_argument('-e', '--experiment', type=str,
                        choices=['relation', 'compression'],
                        default='relation',
                        help='The type of experiment to run. "relation" tests the relation between ' +
                             'inherent linearity and another compression method. "compression" tests ' +
                             'inherent linearity as a tool for compression.')
    parser.add_argument('--relation', type=str,
                        choices=['pruning', 'distillation'],
                        default='pruning',
                        help='The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training and fine-tuning.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for optimizer.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to process during training/evaluation. ' +
                                'If None, process all batches.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    parser.add_argument('--save', action='store_true',
                        help='Save the trained models and results to ./results directory.')
    parser.add_argument('--wandb_project', type=str, default='inherent_linearity_experiments',
                        help='Weights & Biases project name for logging.')
    parser.add_argument('--wandb_api_key',
                        type=str,
                        default=None,
                        help='Your personal API key for Weights and Biases. Default is None. Alternatively, you can ' +
                             'leave this empty and store the key in a file in the root of this script called "wandb.login". ' +
                             'This file will be ignored by git.')
    parser.add_argument('--wandb_tags',
                        type=str,
                        nargs='*',
                        default=[],
                        help='List of tags to add to the Weights and Biases run for better organization.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        filename=f'run-{args.model}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        level= logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    wandb_config = {
        'model': args.model,
        'dataset': args.dataset,
        'experiment': args.experiment,
        'relation': args.relation,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'max_batches': args.max_batches,
    }

    if args.wandb_api_key is not None:
        wandb.login(key=args.wandb_api_key)
    elif os.path.exists('wandb.login'):
        with open('wandb.login', 'r') as f:
            wandb.login(key=f.read())
    else:
        logger.warning("No Weights and Biases API key provided.")

    wandb.init(
        project=args.wandb_project,
        config=wandb_config,
        tags=args.wandb_tags
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info(f"Starting experiment with configuration: {wandb_config}")

    match (args.model, args.experiment, args.relation):
        case ('llama7b' | 'llama13b', 'compression', _):
            from experiments.transformer_compression import run_transformer_compression_experiment
            run_transformer_compression_experiment(args.model, args.dataset, args.batch_size,
                                                   args.epochs, args.lr, args.max_batches, args.save, args.device)
        case _:
            logger.error("Invalid combination of model, experiment, and relation.")
            raise ValueError("Invalid combination of model, experiment, and relation.")