#!/bin/bash

for f in ./run_scripts/rq1/*/resnet/imagenet/*.bash; do
    echo "Submitting $f"
    sbatch "$f"
done