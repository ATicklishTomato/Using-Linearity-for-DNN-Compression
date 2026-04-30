# Thesis Experiments Repository

This repository encompasses experimental code for my thesis on linearity in CNNs and transformers. 
It will eventually contain all experiments and related artifacts.

## Repository layout

- `main.py` \- project entry / orchestration script (see file for CLI usage)
- `sweep.py` \- script for running hyperparameter sweeps using Weights & Biases
- `visualize.py` \- script for summarizing and visualizing results from experiments
- `experiments/` \- general experiment code, which run the experiments and log results
- `compression_methods/` \- code for the different existing compression methods used for comparison and experiments
- `data/` \- directory where datasets are stored or downloaded to
- `metrics/` \- code for the different linearity metrics used in the experiments
- `utils/` \- utility functions for loading datasets, models, finetuning and evaluation, and evaluation metric computation
- `notebook_experiments/` \- Jupyter notebooks with initial exploratory experiments
- `run_scripts/` \- run scripts for Slurm-based execution
- `requirements.txt` \- Python dependencies

## Requirements
Use a virtual environment or conda environment to avoid conflicts with other projects. Python 3.11 is recommended.

Install dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
Installing torch separately before the requirements tends to avoid issues with torch+cuda versioning inside the requirements file.

Place your credentials in the root files:
- Put Hugging Face token in `hf.login`
- Put W&B key in `wandb.login`

## Usage

Below the different ways of running experiments are described.

### Jupyter notebooks

The Jupyter notebooks present only encompass early experimentation and are left to toy around with or provide introductory information.

To run the Jupyter notebooks, use Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook
```

Then navigate to the `notebook_experiments/` directory and open the desired notebook. Make sure to run the cells in order to ensure all dependencies and variables are properly defined.

### Main experimentation

To run an experiment, use the command line interface of `main.py`. This script allows you to specify the model, dataset, linearity metric, and type of experiment you want to run, as well as various training hyperparameters and logging options.

```bash
usage: main.py [-h] [-m {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}] [-l {mean_preactivation,procrustes,fraction}] [-d {imagenet,tinystories,cifar10}]
               [-e {relation,compression,benchmark_compression}] [--relation {magnitude_pruning,basic_kd}] [-t THRESHOLD] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--data_fraction DATA_FRACTION] [--seed SEED]
               [--device DEVICE] [--verbose] [--save] [--wandb_project WANDB_PROJECT] [--wandb_tags [WANDB_TAGS ...]]

Execute experiments on inherent linearity in ResNets and Llamas.

options:
  -h, --help            show this help message and exit
  -m {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}, --model {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}
                        Model architecture to use for the experiment.
  -l {mean_preactivation,procrustes,fraction}, --linearity {mean_preactivation,procrustes,fraction}
                        Linearity metric to use. `mean_preactivation` refers to the mean of preactivations as defined by Pinson et al. (2024). `procrustes` refers to the Procrustes similarity-based metric as defined by
                        Razzhigaev et al (2024). `fraction` refers to the fraction of neurons that is activated by an activation function.
  -d {imagenet,tinystories,cifar10}, --dataset {imagenet,tinystories,cifar10}
                        Dataset to use for training and evaluation.
  -e {relation,compression,benchmark_compression}, --experiment {relation,compression,benchmark_compression}
                        The type of experiment to run. "relation" tests the relation between inherent linearity and another compression method. "compression" tests inherent linearity as a tool for compression.
                        "benchmark_compression" runs other compression methods to allow a comparison.
  --relation {magnitude_pruning,basic_kd}
                        The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.
  -t THRESHOLD, --threshold THRESHOLD
                        The threshold to use for determining what is(n\'t) linear. To take a percentile, enter a percentage, e.g. '75%' to consider anything smaller the 75th percentile as non-linear. To take a hard threshold,
                        enter a floating point value, e.g. '-0.01'. Default is 75th percentile.
  --batch_size BATCH_SIZE
                        Batch size for training and evaluation.
  --epochs EPOCHS       Number of epochs for training and fine-tuning.
  --lr LR               Learning rate for optimizer.
  --data_fraction DATA_FRACTION
                        Fraction of data to use for training and evaluation. If None, default fractions are:- imagenet: 0.1- tinystories: 0.01- cifar10: 1.0
  --seed SEED           Random seed for reproducibility.
  --device DEVICE       Device to run the experiments on (e.g., "cpu", "cuda").
  --verbose             Enable verbose logging.
  --save                Save the trained models and results to ./results directory.
  --wandb_project WANDB_PROJECT
                        Weights & Biases project name for logging.
  --wandb_tags [WANDB_TAGS ...]
                        List of tags to add to the Weights and Biases run for better organization.
```

Example command to run an experiment:

```bash
python main.py -m resnet18 -d imagenet -e compression -t 50%
```

Notes:
- Ensure `hf.login` and `wandb.login` files are present in the root directory with appropriate credentials before running experiments that require them. Alternatively, the W&B API key can be provided directly via the command line argument.
- Ensure you use `--device cuda` if you have a compatible GPU for faster training and evaluation.
- Larger models (e.g., LLama-2-7B) may require significant VRAM. Adjust batch sizes accordingly or use gradient accumulation if necessary.
- Results and trained models will be saved in the `results/` directory if the `--save` flag is used.

### Hyperparameter sweep
A separate script allows for running hyperparameter sweeps using Weights & Biases. See `sweep.py` for details.
Sweeps can be run using `python sweep.py` with appropriate command line arguments to specify the parameters to sweep over.

```bash
usage: sweep.py [-h] [-m {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}] [-l {mean_preactivation,procrustes,fraction}] [-d {imagenet,tinystories}] [-e {relation,compression,benchmark_compression}]
                [--relation {magnitude_pruning,basic_kd}] [-t [THRESHOLD ...]] [--batch_size [BATCH_SIZE ...]] [--epochs [EPOCHS ...]] [--lr [LR ...]] [--data_fraction DATA_FRACTION] [--seed SEED] [--sweep_runs SWEEP_RUNS]
                [--device DEVICE] [--verbose] [--wandb_project WANDB_PROJECT] [--wandb_tags [WANDB_TAGS ...]]

Execute hyperparameter sweep for linearity compression on a specified model, dataset, and linearity metric.

options:
  -h, --help            show this help message and exit
  -m {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}, --model {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}
                        Model architecture to use for the sweep.
  -l {mean_preactivation,procrustes,fraction}, --linearity {mean_preactivation,procrustes,fraction}
                        Linearity metric to use for the sweep. For info on the metrics, check main.py arguments help.
  -d {imagenet,tinystories}, --dataset {imagenet,tinystories}
                        Dataset to use for sweep training and evaluation.
  -e {relation,compression,benchmark_compression}, --experiment {relation,compression,benchmark_compression}
                        The type of experiment to run. "relation" tests the relation between inherent linearity and another compression method. "compression" tests inherent linearity as a tool for compression.
                        "benchmark_compression" runs other compression methods to allow a comparison.
  --relation {magnitude_pruning,basic_kd}
                        The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.
  -t [THRESHOLD ...], --threshold [THRESHOLD ...]
                        The thresholds to try for determining what is(n\'t) linear. To take a percentile, enter a percentage, e.g. '75%' to consider anything smaller the 75th percentile as non-linear. To take a hard threshold,
                        enter a floating point value, e.g. '-0.01'. Default is 75th percentile.
  --batch_size [BATCH_SIZE ...]
                        Batch size for training and evaluation.
  --epochs [EPOCHS ...]
                        Number of epochs for training and fine-tuning.
  --lr [LR ...]         Learning rate for optimizer.
  --data_fraction DATA_FRACTION
                        Fraction of data to use for training and evaluation.
  --seed SEED           Random seed for reproducibility.
  --sweep_runs SWEEP_RUNS
                        Number of runs to execute for the sweep.
  --device DEVICE       Device to run the experiments on (e.g., "cpu", "cuda").
  --verbose             Enable verbose logging.
  --wandb_project WANDB_PROJECT
                        Weights & Biases project name for logging.
  --wandb_tags [WANDB_TAGS ...]
                        List of tags to add to the Weights and Biases run for better organization.
```

### Visualizing average results

A separate script allows for the summarizing of results from different random seeds. Results are grabbed from the `results/` directory and averaged across seeds, then visualized using matplotlib and written to a LaTeX table. See `visualize.py` for details.

```bash
usage: visualize.py [-h] [--rq {rq1,rq2,benchmark}] [--threshold THRESHOLD] [--model {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}] [--dataset {imagenet,tinystories,cifar10}]
                    [--relation_to {magnitude_pruning,basic_kd}] [--linearity {mean_preactivation,procrustes,fraction}]

options:
  -h, --help            show this help message and exit
  --rq {rq1,rq2,benchmark}
                        Which Research Question to aggregate results for
  --threshold THRESHOLD
                        Threshold to aggregate results for
  --model {resnet18,resnet34,resnet50,llama-2-7b,llama-2-13b,llama-3-1b,llama-3-3b}
                        Which model to aggregate results for
  --dataset {imagenet,tinystories,cifar10}
                        Which dataset to aggregate results for
  --relation_to {magnitude_pruning,basic_kd}
                        Which relation type to aggregate results for
  --linearity {mean_preactivation,procrustes,fraction}
                        Linearity metric to use. `mean_preactivation` refers to the mean of preactivations as defined by Pinson et al. (2024). `procrustes` refers to the Procrustes similarity-based metric as defined by
                        Razzhigaev et al (2024). `fraction` refers to the fraction of neurons that is activated by an activation function.
```

## Thanks

I would like to express my thanks to my thesis supervisor, Dr. Hannah Pinson, for her guidance and input throughout the project.
I'd also like to thank everyone in the thesis group of Dr. Pinson for the feedback they gave during our meetings, as well as the pleasant atmosphere at the meetings.

## License
Copyright 2026 Luuk Wubben

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
