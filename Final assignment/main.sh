wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 100 \
    --lr 0.0005 \
    --num-workers 18 \
    --seed 42 \
    --experiment-id "eff + unet-training V2" \
    --dino-fine-tune 0 \
    --dice-weight 0.3 \
    --ce-weight 1 \
    --kd-weight 0.1 \
    --focal-weight 0.1 \
    --accumulation-steps 2 \
    --temperature 4.0