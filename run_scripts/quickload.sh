#!/bin/bash

for f in ./run_scripts/*/fraction/*/resnet/*/*.bash; do
    echo "Submitting $f"
    dos2unix "$f" # Convert to Unix line endings
    sbatch "$f"
    sleep 1 # Sleep for a second to avoid having jobs use the same log file name
done