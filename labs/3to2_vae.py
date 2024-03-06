# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2D Latent Diffusion Model
# In this tutorial, we will walk through the process of using the MONAI Generative Models package to generate synthetic data using Latent Diffusion Models (LDM)  [1, 2]. Specifically, we will focus on training an LDM to create synthetic X-ray images of hands from the MEDNIST dataset.
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
# [2] - Pinaya et al. "Brain imaging generation with latent diffusion models" https://arxiv.org/abs/2209.07162

# ### Setup imports
import os
import glob
import logging
import argparse

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.utils.data import Dataset
from torchsummary import summary
#from torch.cuda.amp import GradScaler, autocast

from monai import transforms
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

nb_acts=0

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--project_name", type=str, default=None, required=True, help="Project name.",)
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data.",)
    parser.add_argument("--train_label_dir", type=str, default=None, help="A folder containing the training label data.",)
    parser.add_argument("--valid_label_dir", type=str, default=None, help="A folder containing the validattion label data.",)
    parser.add_argument("--output_dir", type=str, default="/shared/s1/lab06/wonyoung/GenerativeModels/results/vae-model", help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--input_size", type=str, default=None, help="spatial input size.")
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_steps", type=int, default=100)
    parser.add_argument("--lr_auto", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use for autoencoderkl.",)
    parser.add_argument("--lr_disc", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use for discriminator.",)
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of global steps for the warmup.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--checkpointing_steps", type=int, default=10,)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
    parser.add_argument("--validation_steps", type=int, default=10, help="Run validation every X trainig steps.",)
    #parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    return args

class UKB_Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        super(UKB_Dataset, self).__init__()
        self.data_dir = image_dir
        self.data_csv = pd.read_csv(label_dir)
        self.image_names = list(self.data_csv['id'])
        self.labels = list(self.data_csv['age'])
        # print("images_names: ", len(self.image_names), self.image_names[-1])
        # print("labels: ", len(self.labels), self.labels[-1])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        # Load the image
        image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.float32)
        image = torch.from_numpy(image).float()
        image = image.permute(3, 0, 1, 2) # (1,64,64,64)
        age = torch.tensor(label, dtype=torch.float32)
        sample = dict()
        sample["image"] = image
        sample["label"] = age
        if self.transform:
            sample = self.transform(sample)
            sample["image"] = sample["image"][0,:,:,:] # pseudo color channel (64,64,64)
            # print("sample:", sample["image"])
            # print("sample shape:", sample["image"].shape)

        return sample


def main():
    args = parse_args()
    print_config()
    print("##### args:", args)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = glob.glob(os.path.join(args.output_dir, "*"))
            #dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith(args.output_dir + "/checkpoint")]
            #dirs = [d for d in dirs if d.startswith("checkpoint")]
            #dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            #path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.basename(max(dirs, key=os.path.getctime)) if len(dirs) > 0 else None

    if args.report_to == "wandb": # and args.local_rank == 0:
        # id = wandb.util.generate_id()
        # print("wandb id:", id)
        wandb.init(project=args.project_name, resume=True)
        # if path is None:
        #     wandb.init(project=args.project_name)
        # else:
        #     group_name = path.split("-")[-1]
        #     wandb.init(project=args.project_name, resume=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    print("##### weight_dtype:", weight_dtype)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # for reproducibility purposes set a seed
    if args.seed is not None: # args.seed=42
        set_determinism(args.seed)

    if args.output_dir is not None: # args.output_dir="/shared/s1/lab06/wonyoung/GenerativeModels/results/UKB_test"
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Prepare data loader for the training set
    batch_size = args.train_micro_batch_size_per_gpu
    print("##### batch size per gpu:", batch_size)
    channel = 0  # 0 = Flair # assert channel in [0, 1, 2, 3], "Choose a valid channel"
    input_size = tuple(int(x) for x in args.input_size.split(","))

    train_transforms = transforms.Compose(
            [
                # transforms.LoadImaged(keys=["image"]),
                # transforms.EnsureChannelFirstd(keys=["image"]),
                # transforms.Lambdad(keys="image", func=lambda x: x[channel, :, :, :]),
                # transforms.AddChanneld(keys=["image"]),
                transforms.EnsureTyped(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAS"),
                # transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
                transforms.Resized(keys=["image"], spatial_size=input_size, size_mode="all"),
                # transforms.CenterSpatialCropd(keys=["image"], roi_size=(64, 64, 64)),
                transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1),
            ]
        )

    image_dir = args.train_data_dir # "/leelabsg/data/20252_individual_samples"
    label_dir_train = args.train_label_dir # "/shared/s1/lab06/wonyoung/GenerativeModels/Brain_LDM/ukbb_cn_train.csv" \
    label_dir_valid = args.valid_label_dir # "/shared/s1/lab06/wonyoung/GenerativeModels/Brain_LDM/ukbb_cn_valid.csv" \

    train_ds = UKB_Dataset(image_dir, label_dir_train, transform=train_transforms)
    valid_ds = UKB_Dataset(image_dir, label_dir_valid, transform=train_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

    check_data = first(valid_loader)
    idx = 0

    img = check_data["image"][idx]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    # for ax in axs:
    #     ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    plt.savefig(f"{args.output_dir}/original_examples.png")

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #, args.local_rank)
    print(f"Using {device}")

    # ## Prepare validation set data loader
    # ## Autoencoder KL
    # ### Define Autoencoder KL network, losses and optimiser
    # In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images. We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.
    num_channels = (32, 64, 128, 256)
    autoencoderkl = AutoencoderKL(
        spatial_dims=2,
        in_channels=input_size[0], # depends on input size
        out_channels=input_size[0], # depends on input size
        num_channels=num_channels, # (128, 128, 256),
        latent_channels=int(input_size[0]/(2**(len(num_channels)-1))), # input_size/(2^(4-1))
        num_res_blocks=6,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )
    autoencoderkl = autoencoderkl.to(device, dtype=weight_dtype)

    # autoencoder memory check
    nb_inputs, nb_params = 0, 0
    def count_output_act(m, input, output):
        global nb_acts
        nb_acts += output.nelement()

    for module in autoencoderkl.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.GroupNorm):
            module.register_forward_hook(count_output_act)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    nb_params = sum([p.nelement() for p in autoencoderkl.parameters()])

    autoencoderkl_input = torch.ones(1, input_size[0], input_size[1], input_size[2]).float().to(device) # 
    autoencoderkl.eval() # 13,766,248
    encoded = autoencoderkl(autoencoderkl_input)
    nb_inputs = autoencoderkl_input.nelement()

    print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
        nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
    print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
    print("Autoencdoer parameters:", count_parameters(autoencoderkl))
    summary(autoencoderkl, input_size)

    discriminator = PatchDiscriminator(
        spatial_dims=2, 
        num_layers_d=3, 
        num_channels=96, 
        in_channels=input_size[0], # depends on input size
        out_channels=input_size[0] # depends on input size
    )
    discriminator = discriminator.to(device, dtype=weight_dtype)

    if args.report_to == "wandb":
        wandb.watch(autoencoderkl, log="all")

    # Defining Losses
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    perceptual_loss.to(device)

    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        return torch.sum(kl_loss) / kl_loss.shape[0]
    
    optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=args.lr_auto)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_disc)

    # For mixed precision training
    # scaler_g = torch.cuda.amp.GradScaler()
    # scaler_d = torch.cuda.amp.GradScaler()

    # Load if checkpoint
    if args.resume_from_checkpoint:
        if path is None:
            print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            # client_sd = dict()
            # epoch = 0
            global_step = 0
        else:
            path = os.path.join(args.output_dir, path)
            autoencoder_ckpt = torch.load(f"{path}/autoencoderkl.pth.tar")
            autoencoderkl.load_state_dict(autoencoder_ckpt["state_dict"])
            optimizer_g.load_state_dict(autoencoder_ckpt["optimizer"])

            discriminator_ckpt = torch.load(f"{path}/discriminator.pth.tar")
            discriminator.load_state_dict(discriminator_ckpt["state_dict"])
            optimizer_d.load_state_dict(discriminator_ckpt["optimizer"])
        
            global_step = autoencoder_ckpt["global_step"]
            print(f"Checkpoint '{args.resume_from_checkpoint}'. Start training from global step: {global_step}.")
    else:
        global_step = 0

    # ### Train model
    n_example_images = 4
    adv_weight = 0.005
    perceptual_weight = 0.002
    kl_weight = 0.00000001

    print("[ Start Training ]")
    while global_step < args.num_train_steps:
        print(f"\n ##### Step {global_step+1:3d}: training")
        autoencoderkl.train()
        discriminator.train()
        loss_g_recon, loss_g_kl, loss_g_percep, loss_g_total, gen_loss, disc_loss = 0, 0, 0, 0, 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        
        for step, batch in progress_bar:
            progress_bar.set_description(f"Step {global_step+1}")
            images = batch["image"].to(device, dtype=weight_dtype)
            
            optimizer_g.zero_grad()
            reconstruction, z_mu, z_sigma = autoencoderkl(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float().unsqueeze(dim=1), images.float().unsqueeze(dim=1))
            p_loss = perceptual_loss(reconstruction.float().unsqueeze(dim=1), images.float().unsqueeze(dim=1))
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss
            
            if global_step >= args.warmup_steps:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()
            # scaler_g.scale(loss_g).backward()
            # scaler_g.step(optimizer_g)
            # scaler_g.update()

            if global_step >= args.warmup_steps:
                # Discriminator part
                optimizer_d.zero_grad()
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                # optimizer_d.backward(loss_d)
                # optimizer_d.step()
                optimizer_d.step()

                # scaler_d.scale(loss_d).backward()
                # scaler_d.step(optimizer_d)
                # scaler_d.update()

            loss_g_recon += recons_loss.item()
            loss_g_kl += kl_loss.item()
            loss_g_percep += p_loss.item()
            loss_g_total += loss_g.item()
            if global_step > args.warmup_steps:
                gen_loss += generator_loss.item()
                disc_loss += discriminator_loss.item()

            progress_bar.set_postfix({
                    "recons_loss": loss_g_recon / (step + 1),
                    "loss_g_total": loss_g_total / (step + 1),
                    "gen_loss": gen_loss / (step + 1),
                    "disc_loss": disc_loss / (step + 1),})

            # validation
            if global_step % args.validation_steps == (args.validation_steps - 1):
                print(f"\n ##### Step {global_step+1}: validation")
                autoencoderkl.eval()
                discriminator.eval()
                with torch.no_grad():
                    val_step_loss = 0
                    intermed_imgs = []
                    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), ncols=110)
                    progress_bar.set_description(f"Step {global_step+1}")
                    for step_valid, batch_valid in progress_bar:
                        images = batch_valid["image"].to(device, dtype=weight_dtype)  # choose only one of Brats channels

                        reconstruction, z_mu, z_sigma = autoencoderkl(images)
                        recons_loss = l1_loss(reconstruction.float(), images.float())
                        
                        val_step_loss += recons_loss.item()
                        progress_bar.set_postfix({"val_recons_loss": val_step_loss / (step_valid + 1),})

                        if step_valid < n_example_images:
                            idx = 0
                            img = reconstruction[idx].detach().cpu().numpy()
                            intermed_imgs.append(img)
                    print(f"Validation reconsturction loss: {val_step_loss / (step_valid + 1)}")

            # checkpointing
            if global_step % args.checkpointing_steps == (args.checkpointing_steps - 1):
                print(f"\n ##### Step {global_step+1}: checkpointing")
                os.makedirs(os.path.join(args.output_dir, f"checkpoint-{global_step+1}/intermediary_images"), exist_ok=True)
                for i, img in enumerate(intermed_imgs):
                    fig, axs = plt.subplots(nrows=1, ncols=3)
                    ax = axs[0]
                    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
                    ax = axs[1]
                    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
                    ax = axs[2]
                    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
                    plt.savefig(os.path.join(args.output_dir, f"checkpoint-{global_step+1}/intermediary_images/reconstruction_{i}.png"))
                
                torch.save({"global_step": global_step, 
                            "state_dict": autoencoderkl.state_dict(), 
                            "optimizer" : optimizer_g.state_dict()}, 
                            f"{args.output_dir}/checkpoint-{global_step+1}/autoencoderkl.pth.tar")
                torch.save({"state_dict": discriminator.state_dict(), 
                            "optimizer" : optimizer_d.state_dict()}, 
                            f"{args.output_dir}/checkpoint-{global_step+1}/discriminator.pth.tar")
                print(f"\n ##### Step {global_step+1}: checkpointing done!")

                if args.report_to == "wandb":
                        wandb.log({
                            "Step": global_step + 1,
                            "train_recons_loss": loss_g_recon / (step + 1),
                            "train_kl_loss": loss_g_kl / (step + 1),
                            "train_percep_loss": loss_g_percep / (step + 1),
                            "train_adv_loss": gen_loss / (step + 1),
                            "train_loss_g_total": loss_g_total / (step + 1),
                            "train_disc_loss": disc_loss / (step + 1),
                            "val_recons_loss": val_step_loss / (step_valid + 1),
                        })

            global_step += 1
    
    wandb.finish()

    print("[ End Training ]")
    # del discriminator
    # del loss_perceptual
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

# # ### Visualise the results from the autoencoderKL

# # Plot last 5 evaluations
# val_samples = np.linspace(n_epochs, val_interval, int(n_epochs / val_interval))
# fig, ax = plt.subplots(nrows=5, ncols=1, sharey=True)
# for image_n in range(5):
#     reconstructions = torch.reshape(intermediary_images[image_n], (image_size * num_example_images, image_size)).T
#     ax[image_n].imshow(reconstructions.cpu(), cmap="gray")
#     ax[image_n].set_xticks([])
#     ax[image_n].set_yticks([])
#     ax[image_n].set_ylabel(f"Epoch {val_samples[image_n]:.0f}")

# # ## Diffusion Model
# #
# # ### Define diffusion model and scheduler
# #
# # In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

# # +
# unet = DiffusionModelUNet(
#     spatial_dims=2,
#     in_channels=3,
#     out_channels=3,
#     num_res_blocks=2,
#     num_channels=(128, 256, 512),
#     attention_levels=(False, True, True),
#     num_head_channels=(0, 256, 512),
# )

# scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
# # -

# # ### Scaling factor
# #
# # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
# #
# # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
# #

# # +
# with torch.no_grad():
#     with autocast(enabled=True):
#         z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))

# print(f"Scaling factor set to {1/torch.std(z)}")
# scale_factor = 1 / torch.std(z)
# # -

# # We define the inferer using the scale factor:

# inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# # ### Train diffusion model
# #
# # It takes about ~80 min to train the model.

# # +
# optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

# unet = unet.to(device)
# n_epochs = 200
# val_interval = 40
# epoch_losses = []
# val_losses = []
# scaler = GradScaler()

# for epoch in range(n_epochs):
#     unet.train()
#     autoencoderkl.eval()
#     epoch_loss = 0
#     progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
#     progress_bar.set_description(f"Epoch {epoch}")
#     for step, batch in progress_bar:
#         images = batch["image"].to(device)
#         optimizer.zero_grad(set_to_none=True)
#         with autocast(enabled=True):
#             z_mu, z_sigma = autoencoderkl.encode(images)
#             z = autoencoderkl.sampling(z_mu, z_sigma)
#             noise = torch.randn_like(z).to(device)
#             timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
#             noise_pred = inferer(
#                 inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
#             )
#             loss = F.mse_loss(noise_pred.float(), noise.float())

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         epoch_loss += loss.item()

#         progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
#     epoch_losses.append(epoch_loss / (step + 1))

#     if (epoch + 1) % val_interval == 0:
#         unet.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for val_step, batch in enumerate(val_loader, start=1):
#                 images = batch["image"].to(device)

#                 with autocast(enabled=True):
#                     z_mu, z_sigma = autoencoderkl.encode(images)
#                     z = autoencoderkl.sampling(z_mu, z_sigma)

#                     noise = torch.randn_like(z).to(device)
#                     timesteps = torch.randint(
#                         0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
#                     ).long()
#                     noise_pred = inferer(
#                         inputs=images,
#                         diffusion_model=unet,
#                         noise=noise,
#                         timesteps=timesteps,
#                         autoencoder_model=autoencoderkl,
#                     )

#                     loss = F.mse_loss(noise_pred.float(), noise.float())

#                 val_loss += loss.item()
#         val_loss /= val_step
#         val_losses.append(val_loss)
#         print(f"Epoch {epoch} val loss: {val_loss:.4f}")

#         # Sampling image during training
#         z = torch.randn((1, 3, 16, 16))
#         z = z.to(device)
#         scheduler.set_timesteps(num_inference_steps=1000)
#         with autocast(enabled=True):
#             decoded = inferer.sample(
#                 input_noise=z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl
#             )

#         plt.figure(figsize=(2, 2))
#         plt.style.use("default")
#         plt.imshow(decoded[0, 0].detach().cpu(), vmin=0, vmax=1, cmap="gray")
#         plt.tight_layout()
#         plt.axis("off")
#         plt.show()
# progress_bar.close()

# # -

# # ### Plot learning curves

# plt.figure()
# plt.title("Learning Curves", fontsize=20)
# plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_losses, linewidth=2.0, label="Train")
# plt.plot(
#     np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)), val_losses, linewidth=2.0, label="Validation"
# )
# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.xlabel("Epochs", fontsize=16)
# plt.ylabel("Loss", fontsize=16)
# plt.legend(prop={"size": 14})


# # ### Plotting sampling example
# #
# # Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. For every 100 steps, we store the noisy intermediary samples. In the last step, we decode all latent representations and plot how the image looks like across the sampling process.

# # +
# unet.eval()
# scheduler.set_timesteps(num_inference_steps=1000)
# noise = torch.randn((1, 3, 16, 16))
# noise = noise.to(device)

# with torch.no_grad():
#     image, intermediates = inferer.sample(
#         input_noise=noise,
#         diffusion_model=unet,
#         scheduler=scheduler,
#         save_intermediates=True,
#         intermediate_steps=100,
#         autoencoder_model=autoencoderkl,
#     )


# # -

# # Decode latent representation of the intermediary images
# decoded_images = []
# for image in intermediates:
#     with torch.no_grad():
#         decoded_images.append(image)
# plt.figure(figsize=(10, 12))
# chain = torch.cat(decoded_images, dim=-1)
# plt.style.use("default")
# plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
# plt.tight_layout()
# plt.axis("off")


# # ### Clean-up data directory

# if directory is None:
#     shutil.rmtree(root_dir)
