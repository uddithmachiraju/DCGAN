import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, inChannels, color_channels, outChannels=64):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(

            # Layer 1 - Output: (batch_size, outChannels*16, 4, 4)
            nn.ConvTranspose2d(in_channels=inChannels, 
                               out_channels=outChannels * 16, 
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(outChannels * 16), 
            nn.ReLU(),

            # Layer 2 - Output: (batch_size, outChannels*8, 8, 8)
            nn.ConvTranspose2d(in_channels=outChannels * 16, 
                               out_channels=outChannels * 8, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(outChannels * 8), 
            nn.ReLU(),

            # Layer 3 - Output: (batch_size, outChannels*4, 16, 16)
            nn.ConvTranspose2d(in_channels=outChannels * 8, 
                               out_channels=outChannels * 4, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(outChannels * 4), 
            nn.ReLU(),

            # Layer 4 - Output: (batch_size, outChannels*2, 32, 32)v
            nn.ConvTranspose2d(in_channels=outChannels * 4, 
                               out_channels=outChannels * 2, 
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(outChannels * 2), 
            nn.ReLU(),

            # Layer 5 - Output: (batch_size, outChannels, 64, 64)
            nn.ConvTranspose2d(in_channels=outChannels * 2, 
                               out_channels=color_channels, 
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.layers(input)

# generator = Generator(inChannels=100, color_channels=1)
# # Batch Size, latent Size, Height, Width
# fixed_noise = torch.randn(64, 100, 1, 1) 
# output = generator(fixed_noise) 
# print(output.shape)  # Should give (64, 1, 64, 64)
