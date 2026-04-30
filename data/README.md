# Data directory

This data directory will contain all datasets used for experiments. Below you can find which datasets download and store themselves, and which must be downloaded manually.

| Dataset | Downloaded automatically? | Description |
| --- | --- | --- |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | Yes | A dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class. |
| [ImageNet 2012](http://www.image-net.org/challenges/LSVRC/2012/) | No | A dataset of over 1.2 million images in 1,000 classes, used for large-scale image classification tasks. |
| [TinyStories](https://huggingface.co/datasets/DeepSeek/TinyStories) | Yes | A dataset of 100,000 short stories, each containing 5 sentences, designed for training language models on story generation tasks. |
| [SuperGLUE](https://super.gluebenchmark.com/) | Yes | A benchmark dataset for evaluating the performance of natural language understanding models across a variety of tasks. |

## Data processing
Some datasets will be processed automatically, specifically CIFAR-10 and TinyStories. Others might need manual processing, which is described below.

### ImageNet 2012
The ImageNet 2012 validation data is not structured in a way that allows for easy loading using standard libraries. 
To process the validation data, the recommendation is to download the following script and running it on the validation dataset to structure it into directories for every class:
https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh

