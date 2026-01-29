# Thesis Experiments Repository

This repository collects experiments for the thesis (notebooks, scripts, datasets, results). It will eventually contain all experiments and related artifacts.

## Repository layout

- `main.py` \- project entry / orchestration script (see file for CLI usage)
- `requirements.txt` \- Python dependencies
- `experiments/` \- standalone experiment scripts
- `notebook_experiments/` \- Jupyter notebooks with initial exploratory experiments
- `run_scripts/` \- example run scripts for cluster execution

## Requirements

Install dependencies (Windows):

```bash
python -m pip install -r requirements.txt
```

Use a virtual environment or conda environment.
Place your credentials in the root files:
- Put Hugging Face token in `hf.login`
- Put W&B key in `wandb.login`

## Usage

To run the Jupyter notebooks, use Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook
```

To run an experiment script, use the command line.

```bash
usage: main.py [-h] [-m {resnet18,resnet34,resnet50,llama7b,llama13b}] [-d {imagenet,tinystories}] [-e {relation,compression}] [--relation {pruning,distillation}] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR]
               [--max_batches MAX_BATCHES] [--seed SEED] [--verbose] [--save] [--wandb_project WANDB_PROJECT] [--wandb_api_key WANDB_API_KEY] [--wandb_tags [WANDB_TAGS ...]]

Execute experiments on inherent linearity in ResNets and Llamas.

options:
  -h, --help            show this help message and exit
  -m {resnet18,resnet34,resnet50,llama7b,llama13b}, --model {resnet18,resnet34,resnet50,llama7b,llama13b}
                        Model architecture to use for the experiment.
  -d {imagenet,tinystories}, --dataset {imagenet,tinystories}
                        Dataset to use for training and evaluation.
  -e {relation,compression}, --experiment {relation,compression}
                        The type of experiment to run. "relation" tests the relation between inherent linearity and another compression method. "compression" tests inherent linearity as a tool for compression.
  --relation {pruning,distillation}
                        The relation experiment to run. Only applicable if experiment type is "relation". Ignored otherwise.
  --batch_size BATCH_SIZE
                        Batch size for training and evaluation.
  --epochs EPOCHS       Number of epochs for training and fine-tuning.
  --lr LR               Learning rate for optimizer.
  --max_batches MAX_BATCHES
                        Maximum number of batches to process during training/evaluation. If None, process all batches.
  --seed SEED           Random seed for reproducibility.
  --verbose             Enable verbose logging.
  --save                Save the trained models and results to ./results directory.
  --wandb_project WANDB_PROJECT
                        Weights & Biases project name for logging.
  --wandb_api_key WANDB_API_KEY
                        Your personal API key for Weights and Biases. Default is None. Alternatively, you can leave this empty and store the key in a file in the root of this script called "wandb.login". This file will be
                        ignored by git.
  --wandb_tags [WANDB_TAGS ...]
                        List of tags to add to the Weights and Biases run for better organization.
```

Example command to run an experiment:

```bash
python main.py -m resnet50 -d imagenet -e relation --relation pruning --batch_size 64 --epochs 10 --lr 0.001 --save --wandb_project "thesis_experiments" --wandb_tags resnet50 pruning
```

Notes:
- Ensure `hf.login` and `wandb.login` files are present in the root directory with appropriate credentials before running experiments that require them. Alternatively, the W&B API key can be provided directly via the command line argument.
- Ensure you use `--device cuda` if you have a compatible GPU for faster training and evaluation.
- Larger models (e.g., LLama-2-7B) may require significant VRAM. Adjust batch sizes accordingly or use gradient accumulation if necessary.
- Results and trained models will be saved in the `results/` directory if the `--save` flag is used.

## Thanks

I would like to express my thanks to my thesis supervisor, Dr. Hannah Pinson, for her guidance and input throughout the project.
I'd also like to thank everyone in the thesis group of Dr. Pinson for the feedback they gave during our meetings, as well as the pleasant atmosphere at the meetings.

## License
Copyright 2026 Luuk Wubben

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
