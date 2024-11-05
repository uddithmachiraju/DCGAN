import torch 
import numpy as np
from gtr import Generator 
from trn import train_model  
from disc import Discriminator
import matplotlib.pyplot as plt 
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 

imageSize = 64
batchSize = 1
latentSize = 100 
inputColors = 1 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

transform = transforms.Compose(
    [
        transforms.Resize(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5) 
    ]
)

# Dataset
Data = datasets.FashionMNIST('root', True, transform = transform, download = True) 
trainDataLoader = DataLoader(Data, batch_size = batchSize, shuffle = True)

# Generator and Discriminator
generator = Generator(inChannels=100, color_channels=1) 
discriminator = Discriminator(color_channels = inputColors)

train_model(generator, discriminator, trainDataLoader, 1, batchSize, latentSize, device) 

with torch.no_grad():
    noise = torch.randn(batchSize, latentSize, 1, 1, device = device)
    fakeImage = generator(noise).detach().cpu() 

def displayExampleImages(trainSet, batchSize, title):
    plt.imshow(np.transpose(make_grid(trainSet[:batchSize], padding = 2, normalize = True), (1, 2, 0)))
    plt.axis(False); plt.title(title)
    plt.show()