  python pretrain_flash2.py \
      --epochs 50 \
      --batches-per-epoch 10000 \
      --batch-size 2 \
      --effective-batch-size 32 \
      --lr 3e-4 \
      --save-every 5 \
      --log-interval 100