#!/bin/bash

for f in ./run_scripts/rq1/*/resnet/imagenet/*.bash; do
    echo "Submitting $f"
    sbatch "$f"
    sleep 1 # Sleep for a second to avoid having jobs use the same log file name
done