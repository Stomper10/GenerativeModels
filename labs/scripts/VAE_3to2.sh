#!/bin/bash

#SBATCH --job-name=VAE_3to2_l_res6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
#SBATCH -o /shared/s1/lab06/wonyoung/GenerativeModels/labs/scripts/%x-%j.txt

source /home/s1/wonyoungjang/.bashrc

echo "UKB T1 VAE cognitive normal training."
echo "3d to 2d,
input_size = (160, 224, 160)
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=160, # depends on input size
    out_channels=160, # depends on input size
    num_channels=(128, 256, 256, 256),
    latent_channels=20, # input_size/(2^(4-1))
    num_res_blocks=6,
    attention_levels=(False, False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)"
python3 /shared/s1/lab06/wonyoung/GenerativeModels/labs/3to2_vae.py \
    --project_name=$SLURM_JOB_NAME \
    --train_data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/results/${SLURM_JOB_NAME}" \
    --train_micro_batch_size_per_gpu=4 \
    --seed=42 \
    --input_size="160,224,160" \
    --num_train_steps=100000 \
    --lr_auto=5e-5 \
    --lr_disc=1e-4 \
    --warmup_steps=1000 \
    --mixed_precision="no" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=500 \
    --validation_steps=500 \
    --report_to="wandb" \
