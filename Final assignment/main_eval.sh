#!/bin/bash

echo "Starting Evaluation..."

pip install fvcore

# Run the evaluation script
python3 evaluate.py

echo "Evaluation Finished! Check final_metrics.json"