import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


def progressively_add_noise(img, noise_level):
    return img + np.linspace(0, 1, 10)[noise_level] * torch.randn_like(img)


# Load CIFAR-10 data
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)


class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            128, 3, kernel_size=3, padding=1
        )  # Output channels = 3 for RGB image

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(
            self.conv4(x)
        )  # Using Tanh to keep the output in the same range as the input
        return x


model = DenoiseModel()
print(model)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ask to load first
if Path("denoise_model.pt").exists() and input("load?").lower() == "y":
    model.load_state_dict(torch.load("denoise_model.pt"))

# Pre-generating the noisy dataset
noisy_dataset = []

if not Path("noisy_dataset.pt").exists():
    for images, _ in trainloader:
        for noise_level in range(9):
            noisy_input = progressively_add_noise(images, noise_level)
            next_noisy_target = progressively_add_noise(images, noise_level + 1)
            noisy_dataset.append((noisy_input, next_noisy_target))

    # Save the pre-processed dataset
    torch.save(noisy_dataset, "noisy_dataset.pt")
else:
    # Later, load the dataset for training
    noisy_dataset = torch.load("noisy_dataset.pt")

# Training loop with pre-generated dataset
num_epochs = 10
for epoch in range(num_epochs):
    for noisy_input, next_noisy_target in noisy_dataset:
        optimizer.zero_grad()
        outputs = model(noisy_input)
        loss = criterion(outputs, next_noisy_target)
        loss.backward()
        optimizer.step()

# save
torch.save(model.state_dict(), "denoise_model.pt")

import matplotlib.pyplot as plt


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Get some random training images
dataiter = iter(trainloader)
images, _ = dataiter.next()

# Show original images
imshow(torchvision.utils.make_grid(images))

# Add noise to images
noisy_images = progressively_add_noise(images, 5)  # example noise level

# Show noisy images
imshow(torchvision.utils.make_grid(noisy_images))

# Denoise images
# Remember to set the model to evaluation mode
model.eval()
with torch.no_grad():
    denoised_images = model(noisy_images)

# Show denoised images
imshow(torchvision.utils.make_grid(denoised_images))
