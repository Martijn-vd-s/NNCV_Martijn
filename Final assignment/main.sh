wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 30 \
    --epochs 100 \
    --lr 0.0003 \
    --num-workers 16 \
    --seed 42 \
    --experiment-id "eff + unet-training V1" \
    --dino-fine-tune 0 \
    --dice-weight 0.25 \
    --ce-weight 1 \
    --focal-weight 1 \
    --accumulation-steps 1 \