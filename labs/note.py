# Check model memory usage
import torch.nn as nn
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import logging
import argparse
# import deepspeed
# from deepspeed.ops.adam import DeepSpeedCPUAdam

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
# from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchsummary import summary
# from torchsummary import summary
# from torchvision import transforms as T
# import nibabel as nib
# from skimage.transform import resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoderkl = AutoencoderKL(
        spatial_dims=2,
        in_channels=64,
        out_channels=64,
        num_channels=(128, 256, 256, 512), # (128, 128, 256),
        latent_channels=16, # 128/(2^(4-1))
        num_res_blocks=2,
        attention_levels=(False, False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    )
autoencoderkl.to(device)

nb_acts, nb_inputs, nb_params = 0, 0, 0
def count_output_act(m, input, output):
    global nb_acts
    nb_acts += output.nelement()

for module in autoencoderkl.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.GroupNorm):
        module.register_forward_hook(count_output_act)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

nb_params = sum([p.nelement() for p in autoencoderkl.parameters()])

autoencoderkl_input = torch.ones(1, 64, 64, 64).float().to(device) # 716,800
autoencoderkl.eval() # 13,766,248
encoded = autoencoderkl(autoencoderkl_input)
nb_inputs = autoencoderkl_input.nelement()

print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))
print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))
print("Autoencdoer parameters:", count_parameters(autoencoderkl))
summary(autoencoderkl, (64, 64, 64))


# CNN
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


nb_acts = 0
def count_output_act(m, input, output):
    global nb_acts
    nb_acts += output.nelement()


for module in model_CNN.modules():
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear) or isinstance(module, nn.GroupNorm) or isinstance(module, nn.ReLU):
        module.register_forward_hook(count_output_act)


nb_params = sum([p.nelement() for p in model_CNN.parameters()])
model_CNN_input = torch.ones(1, 1, 128, 128, 128).float().to(device)
model_CNN.eval()
encoded = model_CNN(model_CNN_input)
nb_inputs = model_CNN_input.nelement()

print('input elem: {}, param elem: {}, forward act: {}, mem usage: {}GB'.format(
    nb_inputs, nb_params, nb_acts, (nb_inputs+nb_params+nb_acts)*4/1024**3))

print("{:.2f} GB".format(torch.cuda.memory_allocated() / 1024 ** 3))


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class UKB_Dataset(Dataset):
    def __init__(self):
        super(UKB_Dataset, self).__init__()
        # self.config = config
        self.data_dir = "/leelabsg/data/20252_individual_samples"
        self.data_csv = pd.read_csv("/shared/s1/lab06/wonyoung/GenerativeModels/Brain_LDM/ukbb_cn_small.csv")
        self.image_names = list(self.data_csv['id'])
        self.labels = list(self.data_csv['age'])
        print("images_names: ", len(self.image_names), self.image_names[-1])
        print("labels: ", len(self.labels), self.labels[-1])
        self.transform = T.Compose([
            # T.Resize((image_size, image_size, image_size))
            T.ToTensor()
        ])
    def collate_fn(self, batch):
        images, labels = zip(*batch)  # separate images and labels
        images = torch.stack(images)  # stack images into a tensor
        labels = torch.tensor(labels)  # convert labels into a tensor
        return images, labels
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        # Load the image
        image = np.load(os.path.join(self.data_dir, 'final_array_128_full_' + str(image_name) + '.npy')).astype(np.float32)
        image = torch.from_numpy(image).float()
        image = image.permute(3, 0, 1, 2)
        np.random.seed()
        age = torch.tensor(label, dtype=torch.float32)
        return (image, age)
    
train_dataset = UKB_Dataset()
dataloader_train = DataLoader(train_dataset, 
                            batch_size=1, 
                            sampler=RandomSampler(train_dataset),
                            collate_fn=train_dataset.collate_fn,
                            pin_memory=True,
                            num_workers=8)

check_data = first(dataloader_train)
check_data[0].shape

################# show images

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

file_type = "T1_brain.nii.gz"
path = "/leelabsg/data/20252_test/1000502_20252_2_0/T1"
img = nib.load(os.path.join(path, file_type))
img_data = img.get_fdata()
img_data.shape

test = img_data[:,:,59]
plt.imshow(test)
plt.show()
plt.savefig(f"/shared/s1/lab06/wonyoung/GenerativeModels/Brain_LDM/a_imgs/{file_type}.png")
