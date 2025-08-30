Going down

<img width="1600" height="921" alt="down" src="https://github.com/user-attachments/assets/e6e1f50f-840f-4abc-b694-21957dab9861" />

<img width="1600" height="921" alt="5" src="https://github.com/user-attachments/assets/5b309d34-541b-4ef7-9275-5abda566e82b" />


   
   
   
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


  python pretrain_flash2.py --batch-size 1 --effective-batch-size 8 --lr 1e-5 --resume checkpoints/latest_checkpoint.pt






 python -c "import torch; torch.cuda.empty_cache()"




  watch -n 1 nvidia-smi



    python pretrain_flash2.py \
      --mode inference \
      --model-path checkpoints/latest_checkpoint.pt \
      --prompt "Who are you?" \
      --max-tokens 50 \
      --temperature 0.8

  Interactive Mode (Recommended)

  python pretrain_flash2.py \
      --mode inference \
      --model-path checkpoints/latest_checkpoint.pt \
      --interactive \
      --max-tokens 100 \
      --temperature 0.8

  Creative Writing

  python pretrain_flash2.py \
      --mode inference \
      --model-path checkpoints/latest_checkpoint.pt \
      --prompt "Once upon a time in a distant galaxy" \
      --max-tokens 200 \
      --temperature 1.0

  Code Generation

  python pretrain_flash2.py \
      --mode inference \
      --model-path checkpoints/latest_checkpoint.pt \
      --prompt "def fibonacci(n):" \
      --max-tokens 150 \
      --temperature 0.3







  Continue Training

  python pretrain_flash2.py \
      --resume checkpoints/latest_checkpoint.pt \
      --additional-epochs 10 \
      --batch-size 8 \
      --effective-batch-size 32 \
      --lr 8e-4 \
      --save-every 2 \
      --log-interval 50

  Resume Specific Checkpoint

  python pretrain_flash2.py \
      --resume checkpoints/checkpoint_epoch_0005.pt \
      --additional-epochs 20 \
      --batch-size 8 \
      --effective-batch-size 32 \
      --lr 8e-4 \
      --save-every 2 \
      --log-interval 50

  Resume with Different Settings

  python pretrain_flash2.py \
      --resume checkpoints/latest_checkpoint.pt \
      --additional-epochs 5 \
      --batch-size 4 \
      --effective-batch-size 16 \
      --lr 3e-4 \
      --batches-per-epoch 1000 \
      --save-every 1
