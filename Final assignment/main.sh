wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0005 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "DINOv3 + unet-training V3" \
    --dino-fine-tune 1 \
    --dice-weight 0.25 \
    --ce-weight 1 \
    --focal-weight 1 \