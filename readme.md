   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


   
  
  python pretrain_flash2.py \
      --epochs 10 \
      --batches-per-epoch 2000 \
      --batch-size 16 \
      --effective-batch-size 32 \
      --lr 8e-4 \
      --save-every 2 \
      --log-interval 50



 python -c "import torch; torch.cuda.empty_cache()"



  python pretrain_flash2.py --batch-size 16 ...