#!/bin/bash

source /home/s1/wonyoungjang/.bashrc

echo "UKB T1 VAE cognitive normal training."
python3 /shared/s1/lab06/wonyoung/GenerativeModels/labs/3to2_vae.py \
    --project_name="VAE_3to2_small" \
    --train_data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/results/VAE_3to2_small" \
    --train_micro_batch_size_per_gpu=16 \
    --seed=42 \
    --num_train_steps=100000 \
    --lr_auto=5e-5 \
    --lr_disc=1e-4 \
    --warmup_steps=1000 \
    --mixed_precision="no" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=50 \
    --validation_steps=50 \
    #--report_to="wandb" \
