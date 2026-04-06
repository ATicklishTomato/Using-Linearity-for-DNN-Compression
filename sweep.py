import os
from argparse import ArgumentParser
import logging
from datetime import datetime
import torch
import wandb

# Change wandb logging directory so IDEs don't confuse log directory for importable package
os.environ['WANDB_DIR'] = './wandb_logs'

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(description='Execute hyperparameter sweep for linearity compression on a specified model, dataset, and linearity metric.')
    parser.add_argument('-m', '--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-2-13b', 'llama-3-1b', 'llama-3-3b'],
                        default='resnet18',
                        help='Model architecture to use for the sweep.')
    parser.add_argument('-l', '--linearity', type=str,
                        choices=['mean_preactivation', 'procrustes', 'fraction'],
                        default='mean_preactivation',
                        help='Linearity metric to use for the sweep. For info on the metrics, check main.py arguments help.')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['imagenet', 'tinystories'],
                        default='imagenet',
                        help='Dataset to use for sweep training and evaluation.')
    parser.add_argument('-t', '--threshold', type=str, nargs='*',
                        default=[str(num) + "%" for num in range(25, 90, 5)],
                        help='The thresholds to try for determining what is(n\'t) linear. To take a percentile, ' +
                             'enter a percentage, e.g. \'75%%\' to consider anything smaller the 75th percentile as non-linear. ' +
                             'To take a hard threshold, enter a floating point value, e.g. \'-0.01\'. Default is 75th percentile.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training and fine-tuning.')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate for optimizer.')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum number of batches to process during training/evaluation. ' +
                                'If None, process all batches.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--sweep_runs', type=int, default=20,
                        help='Number of runs to execute for the sweep.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the experiments on (e.g., "cpu", "cuda").')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name for logging.')
    parser.add_argument('--wandb_tags',
                        type=str,
                        nargs='*',
                        default=[],
                        help='List of tags to add to the Weights and Biases run for better organization.')
    return parser.parse_args()

def sweep_train(config_defaults=None):
    generic_args = parse_args()
    wandb.init(config=config_defaults, tags=generic_args.wandb_tags)

    if "llama" in generic_args.model:
        from experiments.llama_approx_compression import run_experiment
        run_experiment(generic_args.model, generic_args.linearity, generic_args.dataset, wandb.config.threshold,
                       wandb.config.batch_size, generic_args.epochs, wandb.config.lr, generic_args.max_batches,
                       False, generic_args.seed, generic_args.device, sweep=True)
    elif "resnet" in generic_args.model:
        from experiments.resnet_fold_compression import run_experiment
        run_experiment(generic_args.model, generic_args.linearity, generic_args.dataset, wandb.config.threshold,
                       wandb.config.batch_size, generic_args.epochs, wandb.config.lr, generic_args.max_batches,
                       False, generic_args.seed, generic_args.device, sweep=True)
    else:
        raise ValueError("Unknown model type.")



if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        filename=f'sweep-{args.model}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        level= logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    if os.path.exists('wandb.login'):
        with open('wandb.login', 'r') as f:
            wandb.login(key=f.read())
    else:
        logger.warning("No Weights and Biases API key provided.")

    project_name = ""
    if args.wandb_project:
        project_name = args.wandb_project
    else:
        project_name = args.model + "_compression"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sweep_config = {
        'method': 'random',
        'metric': {'name': 'compression_score', 'goal': 'maximize'},
        'parameters': {
            'threshold': {'values': args.threshold},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=sweep_train, count=args.sweep_runs)
    wandb.finish()
    logger.info("Sweep complete. Logs saved in sweep.log")

