import torch 
from torch import nn 

class Discriminator(nn.Module):
    def __init__(self, color_channels, outChannels = 64):
        super(Discriminator, self).__init__() 
 
        self.layers = nn.Sequential(
            # Layer1 - Output [64, 1024, 32, 32]
            nn.Conv2d(in_channels = color_channels,
                      out_channels = outChannels * 16, 
                      kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(outChannels * 16),
            nn.ReLU(), 

            # Layer2 - Output [64, 512, 16, 16]
            nn.Conv2d(in_channels = outChannels * 16,
                      out_channels = outChannels * 8, 
                      kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(outChannels * 8),
            nn.ReLU(),

            # Layer3 - Output [64, 256, 8, 8]
            nn.Conv2d(in_channels = outChannels * 8,
                      out_channels = outChannels * 4, 
                      kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(outChannels * 4),
            nn.ReLU(), 

            # Layer4 - Output [64, 128, 4, 4]
            nn.Conv2d(in_channels = outChannels * 4,
                      out_channels = outChannels * 2, 
                      kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(outChannels * 2),
            nn.ReLU(),

            # Layer5 - Output [64, 64, 2, 2]
            nn.Conv2d(in_channels = outChannels * 2,
                      out_channels = outChannels, 
                      kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(), 

            # Layer5 - Output [64, 1, 1, 1]
            nn.Conv2d(in_channels=outChannels, 
                      out_channels=1, 
                      kernel_size=2, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.layers(input)
    
# discriminator = Discriminator(color_channels = 1) 
# noise = torch.randn(64, 1, 64, 64) 
# output = discriminator(noise) 
# print(output.shape)