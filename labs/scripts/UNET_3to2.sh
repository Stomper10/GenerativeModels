#!/bin/bash

#SBATCH --job-name=UNET_3to2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=P2
#SBATCH --time=0-12:00:00
#SBATCH --mem=50GB
#SBATCH --cpus-per-task=8
#SBATCH -o /shared/s1/lab06/wonyoung/GenerativeModels/labs/scripts/%x.txt

source /home/s1/wonyoungjang/.bashrc

echo "UKB T1 UNET cognitive normal training."
echo "3d to 2d,
VAE_3to2_main2,
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=int(input_size[0]/(2**(len(num_channels)-1))), # 3
    out_channels=int(input_size[0]/(2**(len(num_channels)-1))),
    num_res_blocks=4, # 1
    num_channels=(256, 512, 768), # (32, 64, 64)
    attention_levels=(False, True, True),
    num_head_channels=(0, 512, 768),
    with_conditioning=True, ### conditioning
    cross_attention_dim=1024, ### conditioning
)
unet.to(device, dtype=weight_dtype)"
python3 /shared/s1/lab06/wonyoung/GenerativeModels/labs/3to2_unet.py \
    --project_name=$SLURM_JOB_NAME \
    --pretrained_vae_path="/shared/s1/lab06/wonyoung/GenerativeModels/labs/results/VAE_3to2_main2/checkpoint-70000" \
    --train_data_dir="/shared/s1/lab06/20252_individual_samples" \
    --train_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_train.csv" \
    --valid_label_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/data/ukbb_cn_valid.csv" \
    --output_dir="/shared/s1/lab06/wonyoung/GenerativeModels/labs/results/${SLURM_JOB_NAME}" \
    --train_micro_batch_size_per_gpu=4 \
    --seed=42 \
    --input_size="160,224,160" \
    --num_train_steps=100000 \
    --lr_unet=2.5e-05 \
    --mixed_precision="no" \
    --resume_from_checkpoint="latest" \
    --checkpointing_steps=12 \
    --validation_steps=12 \
    #--report_to="wandb" \
