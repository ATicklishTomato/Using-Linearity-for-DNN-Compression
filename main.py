import os
import random
from argparse import ArgumentParser
import logging
from datetime import datetime
import numpy as np
import torch
import wandb
from huggingface_hub import login

# Change wandb logging directory so IDEs don't confuse log directory for importable package
os.environ['WANDB_DIR'] = './wandb_logs'

logger = logging.getLogger(__name__)

def parse_args():
    parser = ArgumentParser(description='Execute experiments on inherent linearity in ResNets and Llamas.')
    parser.add_argument('-m', '--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'llama-2-7b', 'llama-3-1b', 'llama-3-3b'],
                        default='resnet18',
                        help='Model architecture to use for the experiment.')
    parser.add_argument('-l', '--linearity', type=str,
                        choices=['mean_preactivation', 'procrustes', 'fraction'],
                        default='mean_preactivation',
                        help='Linearity metric to use. `mean_preactivation` refers to the mean of preactivations as defined by Pinson et al. (2024). ' +
                             '`procrustes` refers to the Procrustes similarity-based metric as defined by Razzhigaev et al (2024). ' +
                             '`fraction` refers to the fraction of neurons that is activated by an activation function.')
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['imagenet', 'tinystories', 'cifar10', 'superglue'],
                        default='imagenet',
                        help='Dataset to use for training and evaluation.')
    parser.add_argument('-e', '--experiment', type=str,
                        choices=['relation', 'compression', 'linear_approximator_compression', 'benchmark_compression'],
                        default='compression',
                        help='The type of experiment to run. "relation" tests the relation between ' +
                             'inherent linearity and another compression method. "compression" tests ' +
                             'layer merging for ResNets or linear approximation for Llama. ' +
                             '"linear_approximator_compression" tests linear approximation for ResNets. ' +
                             '"benchmark_compression" runs other compression methods to allow a comparison.')
    parser.add_argument('--relation', type=str,
                        choices=['magnitude_pruning', 'basic_kd'],
                        default='magnitude_pruning',
                        help='The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.')
    parser.add_argument('-t', '--threshold', type=str,
                        default=None,
                        help='The threshold to use for determining what is(n\'t) linear. To take a percentile, ' +
                             'enter a percentage, e.g. \'75%%\' to consider anything smaller the 75th percentile as non-linear. ' +
                             'To take a hard threshold, enter a floating point value, e.g. \'-0.01\'. Default is 75th percentile.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training and fine-tuning.')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate for optimizer.')
    parser.add_argument('--data_fraction', type=float, default=None,
                        help='Fraction of data to use for training and evaluation. If None, default fractions are:'
                             '- imagenet: 0.1'
                             '- tinystories: 0.1'
                             '- cifar10: 1.0'
                             '- superglue: 0.1')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the experiments on (e.g., "cpu", "cuda").')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    parser.add_argument('--save', action='store_true',
                        help='Save the trained models and results to ./results directory.')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Set this flag in order to attempt finetune skipping. Instead, the Experimenter class will '
                             'attempt to load a finetuned model from the results directory that matches the model, dataset, and random seed')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name for logging.')
    parser.add_argument('--wandb_tags',
                        type=str,
                        nargs='*',
                        default=[],
                        help='List of tags to add to the Weights and Biases run for better organization.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.data_fraction is None:
        if args.dataset == 'imagenet':
            args.data_fraction = 0.1
        elif args.dataset == 'tinystories':
            args.data_fraction = 0.01
        elif args.dataset == 'cifar10':
            args.data_fraction = 1.0
        elif args.dataset == 'superglue':
            args.data_fraction = 0.1
        else:
            args.data_fraction = 1.0 # fallback

    skip_finetune_path = None
    if args.skip_finetune and "llama" in args.model:
        skip_finetune_path = f"./results/**/{args.model}/{args.dataset}/{args.seed}/original_{args.model}"
    elif args.skip_finetune:
        skip_finetune_path = f"./results/**/{args.model}/{args.dataset}/{args.seed}/{args.model}_original.pth"


    logging.basicConfig(
        filename=f'run-{args.model}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log',
        level= logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(asctime)s (%(filename)s, %(funcName)s) - %(message)s"
    )

    wandb_config = {
        'model': args.model,
        'dataset': args.dataset,
        'experiment': args.experiment,
        'linearity': args.linearity,
        'threshold': args.threshold,
        'relation': args.relation,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'data_fraction': args.data_fraction,
        'skip_finetune': args.skip_finetune
    }

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
    else:
        project_name = args.model + "_" + args.experiment

    wandb.init(
        entity="linearity-thesis",
        project=project_name,
        config=wandb_config,
        tags=args.wandb_tags
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info(f"Starting experiment with configuration: {wandb_config}")

    match (args.model, args.experiment):
        case ('llama-2-7b' | 'llama-3-1b' | 'llama-3-3b', 'compression' | 'linear_approximator_compression'):
            from experiments.llama_approx_compression import run_experiment
            run_experiment(args.model, args.linearity, args.dataset, args.threshold, args.batch_size,
                           args.epochs, args.lr, args.data_fraction, args.save, args.seed, args.device,
                           skip_finetune_path)
        case ('resnet18' | 'resnet34' | 'resnet50', 'compression'):
            from experiments.resnet_fold_compression import run_experiment
            run_experiment(args.model, args.linearity, args.dataset, args.threshold, args.batch_size,
                           args.epochs, args.lr, args.data_fraction, args.save, args.seed, args.device,
                           skip_finetune_path)
        case ('resnet18' | 'resnet34' | 'resnet50', 'linear_approximator_compression'):
            from experiments.resnet_approx_compression import run_experiment
            run_experiment(args.model, args.linearity, args.dataset, args.threshold, args.batch_size,
                           args.epochs, args.lr, args.data_fraction, args.save, args.seed, args.device,
                           skip_finetune_path)
        case (_, 'benchmark_compression'):
            from experiments.benchmark_compression import run_experiment
            run_experiment(args.model, args.dataset, args.batch_size, args.epochs, args.lr,
                           args.data_fraction, args.save, args.seed, args.device,
                           skip_finetune_path)
        case (_, 'relation'):
            from experiments.relation import run_experiment
            run_experiment(args.model, args.linearity, args.dataset, args.relation, args.batch_size, args.epochs, args.lr,
                                          args.data_fraction, args.save, args.seed, args.device, skip_finetune_path)
        case _:
            logger.error("Invalid combination of model, experiment, and relation.")
            raise ValueError("Invalid combination of model, experiment, and relation.")