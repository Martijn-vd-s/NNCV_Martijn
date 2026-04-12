#!/bin/bash

echo "Starting Evaluation..."

pip install git+https://github.com/lucasb-eyer/pydensecrf.git

# Run the evaluation script
python3 evaluate.py

echo "Evaluation Finished! Check final_metrics.json"