#!/bin/bash

for f in ./run_scripts/*/*/*/resnet/*/*.bash; do
    echo "Submitting $f"
    sbatch "$f"
    sleep 1 # Sleep for a second to avoid having jobs use the same log file name
done