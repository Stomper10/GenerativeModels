#!/bin/bash

#SBATCH --job-name=VAE_3to2_main4
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
num_channels = (256, 128, 64, 32)
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=input_size[0], # depends on input size
    out_channels=input_size[0], # depends on input size
    num_channels=num_channels, # (128, 128, 256),
    latent_channels=int(input_size[0]/(2**(len(num_channels)-1))), # input_size/(2^(4-1))
    num_res_blocks=4,
    attention_levels=tuple(False for _ in num_channels),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
adv_weight = 0.005
perceptual_weight = 0.002
kl_weight = 0.00000001
ffl3d_weight = 20
"
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
