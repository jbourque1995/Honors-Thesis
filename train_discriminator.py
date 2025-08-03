import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import os

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(8 * 8 * 128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train_discriminator():
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = Discriminator().cuda()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(5):
        for images, _ in trainloader:
            real = images.cuda()
            label_real = torch.ones(real.size(0), 1).cuda()
            label_fake = torch.zeros(real.size(0), 1).cuda()

            fake = torch.randn_like(real).cuda()  # Placeholder for fake input

            optimizer.zero_grad()
            output_real = model(real)
            loss_real = criterion(output_real, label_real)

            output_fake = model(fake)
            loss_fake = criterion(output_fake, label_fake)

            loss = loss_real + loss_fake
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/5 complete")

    torch.save(model.state_dict(), "/content/drive/MyDrive/Honors_Thesis/discriminator.pth")
    print("Discriminator saved to discriminator.pth")

if __name__ == "__main__":
    train_discriminator()
