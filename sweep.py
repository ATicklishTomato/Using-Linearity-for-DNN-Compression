import os
from argparse import ArgumentParser
import logging
from datetime import datetime
import torch
import wandb
from huggingface_hub import login

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
    parser.add_argument('-e', '--experiment', type=str,
                        choices=['relation', 'compression', 'benchmark_compression'],
                        default='compression',
                        help='The type of experiment to run. "relation" tests the relation between ' +
                             'inherent linearity and another compression method. "compression" tests ' +
                             'inherent linearity as a tool for compression. "benchmark_compression" runs other compression methods to allow a comparison.')
    parser.add_argument('--relation', type=str,
                        choices=['magnitude_pruning', 'basic_kd'],
                        default='magnitude_pruning',
                        help='The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.')
    parser.add_argument('-t', '--threshold', type=str, nargs='*',
                        default=["50%"],
                        help='The thresholds to try for determining what is(n\'t) linear. To take a percentile, ' +
                             'enter a percentage, e.g. \'75%%\' to consider anything smaller the 75th percentile as non-linear. ' +
                             'To take a hard threshold, enter a floating point value, e.g. \'-0.01\'. Default is 75th percentile.')
    parser.add_argument('--batch_size', type=int, default=[64,128,256], nargs='*',
                        help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=[10,20,30], nargs='*',
                        help='Number of epochs for training and fine-tuning.')
    parser.add_argument('--lr', type=float, default=[2e-4], nargs='*',
                        help='Learning rate for optimizer.')
    parser.add_argument('--data_fraction', type=float, default=0.05,
                        help='Fraction of data to use for training and evaluation.')
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
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 2e-5,
        "epochs": 10,
        "threshold": "50%",
        "pruning_ratio": 0.1,
        "blocks": None,
        "hidden_layer_reduction": 2,
    }
    generic_args = parse_args()
    wandb.init(config=config_defaults, tags=generic_args.wandb_tags)

    if "llama" in generic_args.model and generic_args.experiment == "compression":
        from experiments.llama_approx_compression import run_experiment
        run_experiment(generic_args.model, generic_args.linearity, generic_args.dataset, wandb.config.threshold,
                       generic_args.batch_size, generic_args.epochs, generic_args.lr, generic_args.data_fraction,
                       False, generic_args.seed, generic_args.device, sweep=True)
    elif "resnet" in generic_args.model and generic_args.experiment == "compression":
        from experiments.resnet_fold_compression import run_experiment
        run_experiment(generic_args.model, generic_args.linearity, generic_args.dataset, wandb.config.threshold,
                       generic_args.batch_size, generic_args.epochs, generic_args.lr, generic_args.data_fraction,
                       False, generic_args.seed, generic_args.device, sweep=True)
    elif generic_args.experiment == "relation":
        from experiments.relation import run_experiment
        run_experiment(generic_args.model, generic_args.linearity, generic_args.dataset, generic_args.relation, wandb.config.batch_size, wandb.config.epochs,
                       wandb.config.lr, generic_args.data_fraction, False, generic_args.seed, generic_args.device,
                       pruning_ratio=wandb.config.pruning_ratio, blocks=wandb.config.blocks, hidden_layer_reduction=wandb.config.hidden_layer_reduction)
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
            os.environ['WANDB_API_KEY'] = f.read().strip()
    else:
        logger.warning("No Weights and Biases API key provided.")

    if os.path.exists('hf.login'):
        login(token=open("hf.login").read().strip())
    else:
        logger.warning("No HuggingFace API key provided.")

    project_name = ""
    if args.wandb_project:
        project_name = args.wandb_project
    elif args.experiment == "compression":
        project_name = args.model + "_compression"
    elif args.experiment == "relation":
        project_name = args.model + "_relation"
    else:
        raise ValueError("Unknown experiment type.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.experiment == "compression":
        target = {'name': 'compression_score', 'goal': 'maximize'}
    else:
        target = {'name': 'comp_acc', 'goal': 'maximize'}

    sweep_config = {
        'method': 'random',
        'metric': target,
        'parameters': {
            'epochs': {'values': args.epochs},
            'lr': {'values': args.lr},
            'batch_size': {'values': args.batch_size},
        }
    }

    if args.experiment == "compression":
        sweep_config['parameters']['threshold'] = {'values': args.threshold}
    elif args.experiment == "relation":
        if args.relation in ['magnitude_pruning']:
            sweep_config['parameters']['pruning_ratio'] = {'values': [0.1, 0.2, 0.3, 0.4]}
        else:
            if "resnet" in args.model:
                sweep_config['parameters']['blocks'] = {'values': [
                    [1, 2, 2, 2],
                    [1, 1, 2, 2],
                    [1, 1, 1, 2]
                ]}
            else:
                sweep_config['parameters']['hidden_layer_reduction'] = {'values': [2, 3, 4, 5]}

    sweep_id = wandb.sweep(sweep_config, entity="linearity-thesis", project=project_name)
    wandb.agent(sweep_id, function=sweep_train, count=args.sweep_runs)
    wandb.finish()
    logger.info("Sweep complete. Logs saved in sweep.log")

