import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from agent.agent import RESNET_VAE, VAE
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

class PongDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_list[idx])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Hyper parameters
num_epochs = 20
lr = 0.0001

BATCH_SIZE = 32
TRAIN_DATA_PATH = "dataset/train"
TEST_DATA_PATH = "dataset/test"
TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor()
    ])
RESIZE = transforms.Resize([256, 256])

train_data = PongDataset(root_dir=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = PongDataset(root_dir=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True) 

# vae = VAE(image_channels=3, h_dim=50176, z_dim=512)
vae = RESNET_VAE(3, 128, 512)

# Loss and Optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=lr) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)


# Train the Model
for epoch in range(num_epochs):
    for idx, images in enumerate(train_data_loader):
        images = RESIZE(images)
        recon_images, mu, logvar = vae(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        # if idx % 50 == 0:
        #     io.imshow(recon_images.detach()[0].permute(1, 2, 0).numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        to_print = " Epoch[{}/{}] Batch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                num_epochs, idx+1, len(train_data_loader), loss/BATCH_SIZE, bce/BATCH_SIZE, kld/BATCH_SIZE)
        print(to_print)

        #Save the Trained Model
        torch.save(vae.state_dict(), 'vae_{}.pkl'.format(epoch))
    # Test the Model
    with torch.no_grad():
        loss_ = 0.0
        bce_ = 0.0
        kld_ = 0.0
        for idx, images in enumerate(train_data_loader):
            images = RESIZE(images)
            recon_images, mu, logvar = vae(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ += loss
            bce_ += bce
            kld_ += kld

        to_print = "Test Results: Loss: {:.3f} {:.3f} {:.3f}".format(loss_/len(test_data_loader), bce_/len(test_data_loader), kld_/len(test_data_loader))
        print(to_print)

#Save the Trained Model
torch.save(vae.state_dict(), 'vae_final.pkl')
