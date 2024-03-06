import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism

from torch.utils.data import Dataset ###
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

print_config()
# -

# for reproducibility purposes set a seed
set_determinism(42)

# ### Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# ### Prepare data loader for the training set
# Here we will download the Brats dataset using MONAI's `DecathlonDataset` class, and we prepare the data loader for the training set.

# +
batch_size = 1 ###
weight_dtype = torch.float32 ###
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(2.4, 2.4, 2.2), mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=(160, 224, 160)),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.5, b_min=0, b_max=1),
    ]
)

class Dummy(Dataset):
    def __init__(self, input_size, data_length, transform=None):
        super(Dummy, self).__init__()
        self.input_size = input_size
        self.len = data_length
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = torch.randn(1, self.input_size[0], self.input_size[1], self.input_size[2], dtype=weight_dtype)
        label = torch.randn(1, 4, dtype=weight_dtype)
        sample = dict()
        sample["image"] = image
        sample["label"] = label
        if self.transform:
            sample = self.transform(sample)

        return sample

train_ds = Dummy(input_size=(80, 112, 80), data_length=100000, transform=None)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
print(f'Image shape {train_ds[0]["image"].shape}')
# -

# ### Visualise examples from the training set

# +
# Plot axial, coronal and sagittal slices of a training sample
check_data = first(train_loader)
idx = 0

img = check_data["image"][idx, 0]
fig, axs = plt.subplots(nrows=1, ncols=3)
for ax in axs:
    ax.axis("off")
ax = axs[0]
ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
ax = axs[1]
ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
ax = axs[2]
ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
plt.savefig("/shared/s1/lab06/wonyoung/GenerativeModels/labs/original_examples.png")
# plt.savefig("training_examples.png")
# -

# ## Autoencoder KL
#
# ### Define Autoencoder KL network
#
# In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# +
autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=3,
    num_channels=(
        64,
        128,
        128,
        128
    ),
    num_res_blocks=2,
    attention_levels=(
        False,
        False,
        False,
        False
    ),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False
)
autoencoder.to(device, dtype=weight_dtype) ###
autoencoder.load_state_dict(torch.load("/shared/s1/lab06/wonyoung/GenerativeModels/model-zoo/models/brain_image_synthesis_latent_diffusion_model/models/autoencoder.pth"))

########## Model check ##########
import torch
import torch.nn as nn
from torchsummary import summary

# autoencoder memory check
nb_acts, nb_inputs, nb_params = 0, 0, 0
def count_output_act(m, input, output):
    global nb_acts
    nb_acts += output.nelement()

for module in autoencoder.modules():
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.GroupNorm):
        module.register_forward_hook(count_output_act)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nb_params = sum([p.nelement() for p in autoencoder.parameters()])

autoencoder_input = torch.ones(1, 1, 80, 112, 80).float().to(device) # 716,800
autoencoder.eval() # 13,766,248
encoded = autoencoder(autoencoder_input)
nb_inputs = autoencoder_input.nelement()

print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
    nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
print("Autoencdoer parameters:", count_parameters(autoencoder))
summary(autoencoder, (1, 80, 112, 80))



# 3d CNN
class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.MaxPool3d(kernel_size=(2,2,2))
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.MaxPool3d(2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96),
            nn.Conv3d(96, 96, kernel_size=(3,3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm3d(96)
        )
        
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(768, 96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

model_CNN = CNN(in_channels=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_CNN.to(device)

nb_acts, nb_inputs, nb_params = 0, 0, 0
def count_output_act(m, input, output):
    global nb_acts
    nb_acts += output.nelement()

for module in model_CNN.modules():
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm3d):
        module.register_forward_hook(count_output_act)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nb_params = sum([p.nelement() for p in model_CNN.parameters()])

model_CNN_input = torch.ones(1, 1, 128, 128, 128).float().to(device) # 
model_CNN.eval() # 
encoded = model_CNN(model_CNN_input)
nb_inputs = model_CNN_input.nelement()

print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
    nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
print("model_CNN parameters:", count_parameters(model_CNN))
summary(model_CNN, (1, 128, 128, 128))




# densenet169 memory check (2d vs 3d comparison)
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=True)
model.to(device, dtype=weight_dtype)

nb_acts, nb_inputs, nb_params = 0, 0, 0
def count_output_act(m, input, output):
    global nb_acts
    nb_acts += output.nelement()

for module in model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
        module.register_forward_hook(count_output_act)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
nb_params = sum([p.nelement() for p in model.parameters()])

model_input = torch.ones(1, 3, 1280, 1280).float().to(device) # 6,220,800
model.eval() # 14,149,480
encoded = model(model_input)
nb_inputs = model_input.nelement()

print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
    nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
print("densenet169 parameters:", count_parameters(model))
summary(model, (3, 1600, 1600), batch_size=1)
########## Model check ##########

discriminator = PatchDiscriminator(
    spatial_dims=3, 
    num_layers_d=3, 
    num_channels=96, ###
    in_channels=1, 
    out_channels=1)
discriminator.to(device, dtype=weight_dtype) ###
# -

# ### Defining Losses
#
# We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

# +
l1_loss = L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.25) ###
loss_perceptual.to(device)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


adv_weight = 0.005 # 0.01
perceptual_weight = 0.002 # 0.001
kl_weight = 1e-8 # 1e-6
# -

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=5e-5) # lr=1e-4
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)  # lr=1e-4

# ### Train model

# +
n_epochs = 20 # 100
autoencoder_warm_up_n_epochs = 5 # 5
val_interval = 10
epoch_recon_loss_list = []
epoch_gen_loss_list = []
epoch_disc_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

print("[ Start Training ]")
for epoch in range(n_epochs):
    print(f"\n ##### Epoch {epoch+1:3d}: training")
    autoencoder.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device, dtype=weight_dtype)  # choose only one of Brats channels ###

        # Generator part
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)
        kl_loss = KL_loss(z_mu, z_sigma)

        recons_loss = l1_loss(reconstruction.float(), images.float())
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        if epoch > autoencoder_warm_up_n_epochs:
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

del discriminator
del loss_perceptual
torch.cuda.empty_cache()
