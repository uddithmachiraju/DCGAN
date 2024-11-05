import torch 
from tqdm import tqdm 
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_discriminator(discriminator, generator, 
                        realImages, D_optimizer, batchSize, 
                        latentSize, device):
    D_optimizer.zero_grad() 

    # Pass real labels 
    realPreds = discriminator(realImages).view(-1) 
    realLabels = torch.ones(batchSize, device = device)
    realLoss = F.binary_cross_entropy(realPreds, realLabels) 
    realScore = realLoss.mean().item() 

    # Generate Fake data using Generator
    noise = torch.randn(batchSize, latentSize, 1, 1, device = device) 
    generatedImage = generator(noise) 

    # Pass Fake data to discriminator
    fakePreds = discriminator(generatedImage).view(-1)
    fakeLabels = torch.zeros(batchSize, device = device)
    fakeLoss = F.binary_cross_entropy(fakePreds, fakeLabels)
    fakeScore = fakeLoss.mean().item() 

    # Update loss and Weights
    loss = realLoss + fakeLoss 
    loss.backward() 
    D_optimizer.step() 

    return loss.item(), realScore, fakeScore

def train_generator(generator, discriminator, G_optimizer, 
                    batchSize, latentSize, device):
    G_optimizer.zero_grad() 

    # Generate Noise
    noise = torch.randn(batchSize, latentSize, 1, 1, device = device)
    generatedImage = generator(noise) 

    # Fool the discriminator
    preds = discriminator(generatedImage).view(-1)
    labels = torch.ones(batchSize, device = device)
    loss = F.binary_cross_entropy(preds, labels)

    # Update the loss and Weights
    loss.backward()
    G_optimizer.step()

    return loss.item()

def train_model(generator, discriminator, data_loader, 
                epochs, batchSize, latentSize, device):
    D_loss = []
    G_loss = []
    realScore = []
    fakeScore = []

    G_optimizer = torch.optim.Adam(params = generator.parameters(), lr = 0.002, betas = (0.5, 0.999))
    D_optimizer = torch.optim.Adam(params = discriminator.parameters(), lr = 0.002, betas = (0.5, 0.999))

    for epoch in range(epochs):
        for image, _ in tqdm(data_loader):
            image = image.to(device = device)
            d_loss, real_score, fake_score = train_discriminator(discriminator, generator, image, 
                                                                 D_optimizer, batchSize, latentSize, device)
            g_loss = train_generator(generator, discriminator, G_optimizer, 
                                     batchSize, latentSize, device)
        D_loss.append(d_loss) 
        G_loss.append(g_loss)

        realScore.append(real_score)
        fakeScore.append(fake_score)

        if epoch % 1 == 0:
            print(f'Epoch: {epoch}, DiscriminatorLoss: {d_loss}, GeneratorLoss: {g_loss}')
            print(f'Real Score: {real_score}, Fake Score: {fake_score}')
            print()

    plot(D_loss, G_loss, realScore, fakeScore)

def plot(D_loss, G_loss, realScore, fakeScore):
    epochs = len(D_loss)

    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), D_loss, label="Discriminator Loss")
    plt.plot(range(epochs), G_loss, label="Generator Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Discriminator and Generator Losses')

    # Plot scores
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), realScore, label="Real Score")
    plt.plot(range(epochs), fakeScore, label="Fake Score")
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Real and Fake Scores')

    plt.tight_layout()
    plt.show()

