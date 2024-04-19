# export CUDA_VISIBLE_DEVICES=0

accelerate launch --mixed_precision='bf16' \
    train.gemma.infini.noclm.py \
    --model_name_or_path='google/gemma-2b' \
    --segment_length=2048 \
    --block_size=32768 \
    --dataset_name='JeanKaddour/minipile' \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --output_dir='./models/gemma-2b-infini-noclm-minipile' \
    --checkpointing_steps=100 \
    --num_train_epochs=1 \
    --learning_rate=5e-5 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=64 \
    --with_tracking \
