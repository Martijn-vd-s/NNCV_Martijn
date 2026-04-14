wandb login

pip install git+https://github.com/mit-han-lab/efficientvit.git

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.0005 \
    --num-workers 18 \
    --seed 42 \
    --experiment-id "eff + unet-training V2.2" \
    --dice-weight 1.0 \
    --ce-weight 1 \
    --focal-weight 0 \
    --accumulation-steps 2 \
    --temperature 4.0