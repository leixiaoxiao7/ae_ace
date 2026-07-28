import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transform to convert images to tensors
tensor_transform = transforms.ToTensor()

# Load MNIST training dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensor_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Load MNIST test dataset (optional, for evaluation)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=tensor_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # Latent space dimension
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)  # Flatten image
        
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example visualization of reconstructed images
def visualize_reconstructions(model, data_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i * images.size(0) >= num_images:
                break
            
            original_images = images[:num_images].view(-1, 28*28)
            reconstructed_images = model(original_images).view(-1, 28, 28)
            
            plt.figure(figsize=(10, 4))
            for j in range(num_images):
                plt.subplot(2, num_images, j + 1)
                plt.imshow(images[j].squeeze(), cmap='gray')
                plt.axis('off')
                
                plt.subplot(2, num_images, j + num_images + 1)
                plt.imshow(reconstructed_images[j].squeeze(), cmap='gray')
                plt.axis('off')
            plt.show()
            break

visualize_reconstructions(model, test_loader) 