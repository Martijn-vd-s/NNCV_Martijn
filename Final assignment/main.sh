wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \ #16 was too big
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "DINOv3 + unet-training V3" \
    --dino-fine-tune 1 \