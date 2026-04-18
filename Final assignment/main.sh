wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.0002 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segb5-unet-V6" \
    --dino-fine-tune true \
    --dice-weight 0.25 \
    --ce-weight 1 \
    --focal-weight 1 \
    --accumulation-steps 2