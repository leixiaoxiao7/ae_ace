import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

class RegionSelectingAutoencoder(nn.Module):
    """
    Autoencoder that learns to select important regions while keeping features together.
    Uses a bottleneck layer with region-wise weights.
    """
    def __init__(self, n_regions=165, n_features=5, n_selected_regions=15):
        super(RegionSelectingAutoencoder, self).__init__()
        self.n_regions = n_regions
        self.n_features = n_features
        self.n_selected_regions = n_selected_regions
        
        # Encoder layers
        self.encoder_fc1 = nn.Linear(n_regions * n_features, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        
        # Region importance weights (learnable)
        # This will learn the importance of each region
        self.region_weights = nn.Linear(256, n_regions)
        
        # Bottleneck projection
        self.bottleneck = nn.Linear(256, n_selected_regions * n_features)
        
        # Decoder layers
        self.decoder_fc1 = nn.Linear(n_selected_regions * n_features, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_fc3 = nn.Linear(512, n_regions * n_features)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def encode(self, x):
        # Flatten the input while keeping batch dimension
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Encode
        h1 = self.relu(self.encoder_fc1(x))
        h2 = self.relu(self.encoder_fc2(h1))
        
        # Get region importance scores
        region_scores = self.region_weights(h2)
        region_attention = self.softmax(region_scores)
        
        # Apply attention and create bottleneck
        bottleneck = self.relu(self.bottleneck(h2))
        bottleneck = bottleneck.view(batch_size, self.n_selected_regions, self.n_features)
        
        return bottleneck, region_attention
    
    def decode(self, z):
        batch_size = z.shape[0]
        z = z.view(batch_size, -1)
        
        h1 = self.relu(self.decoder_fc1(z))
        h2 = self.relu(self.decoder_fc2(h1))
        reconstruction = self.decoder_fc3(h2)
        
        # Reshape to original dimensions
        reconstruction = reconstruction.view(batch_size, self.n_regions, self.n_features)
        return reconstruction
    
    def forward(self, x):
        bottleneck, region_attention = self.encode(x)
        reconstruction = self.decode(bottleneck)
        return reconstruction, bottleneck, region_attention

class RegionSelector:
    """
    Main class for region selection using autoencoder
    """
    def __init__(self, n_regions=165, n_features=5, n_selected_regions=15):
        self.n_regions = n_regions
        self.n_features = n_features
        self.n_selected_regions = n_selected_regions
        self.model = None
        self.region_importance = None
        self.selected_indices = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, X, batch_size=32, validation_split=0.2):
        """
        Prepare data for training
        X: numpy array of shape (n_subjects, n_regions, n_features)
        """
        # Normalize the data
        n_subjects = X.shape[0]
        X_flat = X.reshape(n_subjects, -1)
        X_normalized = self.scaler.fit_transform(X_flat)
        X_normalized = X_normalized.reshape(n_subjects, self.n_regions, self.n_features)
        
        # Split into train and validation
        n_train = int(n_subjects * (1 - validation_split))
        indices = np.random.permutation(n_subjects)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X_normalized[train_indices]
        X_val = X_normalized[val_indices]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, X_normalized
    
    def train(self, X, epochs=100, learning_rate=0.001, batch_size=32, 
              regularization_weight=0.01, device='cpu'):
        """
        Train the autoencoder for region selection
        """
        # Initialize model
        self.model = RegionSelectingAutoencoder(
            self.n_regions, self.n_features, self.n_selected_regions
        ).to(device)
        
        # Prepare data
        train_loader, val_loader, X_normalized = self.prepare_data(X, batch_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        val_losses = []
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                
                # Forward pass
                reconstruction, bottleneck, region_attention = self.model(batch_data)
                
                # Calculate losses
                recon_loss = criterion(reconstruction, batch_target)
                
                # Add L1 regularization on region attention for sparsity
                sparsity_loss = regularization_weight * torch.mean(torch.abs(region_attention))
                
                total_loss = recon_loss + sparsity_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data = batch_data.to(device)
                    batch_target = batch_target.to(device)
                    
                    reconstruction, _, _ = self.model(batch_data)
                    loss = criterion(reconstruction, batch_target)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        # Calculate region importance scores
        self._calculate_region_importance(X_normalized, device)
        
        return train_losses, val_losses
    
    def _calculate_region_importance(self, X_normalized, device):
        """
        Calculate importance scores for each region
        """
        self.model.eval()
        all_attention_weights = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(device)
            # Process in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                _, _, attention = self.model(batch)
                all_attention_weights.append(attention.cpu().numpy())
        
        # Aggregate attention weights across all samples
        all_attention_weights = np.concatenate(all_attention_weights, axis=0)
        self.region_importance = np.mean(all_attention_weights, axis=0)
        
        # Select top regions based on importance
        self.selected_indices = np.argsort(self.region_importance)[-self.n_selected_regions:]
        self.selected_indices = np.sort(self.selected_indices)
        
        print(f"\nTop {self.n_selected_regions} selected regions (indices): {self.selected_indices}")
    
    def transform(self, X):
        """
        Transform the input data to selected regions only
        X: numpy array of shape (n_subjects, n_regions, n_features)
        Returns: array of shape (n_subjects, n_selected_regions, n_features)
        """
        if self.selected_indices is None:
            raise ValueError("Model must be trained first. Call train() before transform().")
        
        # Select the important regions
        X_selected = X[:, self.selected_indices, :]
        return X_selected
    
    def get_reduced_representation(self, X, device='cpu'):
        """
        Get the bottleneck representation from the autoencoder
        """
        if self.model is None:
            raise ValueError("Model must be trained first.")
        
        self.model.eval()
        
        # Normalize the data
        n_subjects = X.shape[0]
        X_flat = X.reshape(n_subjects, -1)
        X_normalized = self.scaler.transform(X_flat)
        X_normalized = X_normalized.reshape(n_subjects, self.n_regions, self.n_features)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_normalized).to(device)
            bottleneck, _ = self.model.encode(X_tensor)
            return bottleneck.cpu().numpy()
    
    def plot_training_history(self, train_losses, val_losses):
        """
        Plot training and validation losses
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_region_importance(self):
        """
        Plot region importance scores
        """
        if self.region_importance is None:
            raise ValueError("Model must be trained first.")
        
        plt.figure(figsize=(12, 6))
        
        # Plot all region importance scores
        plt.subplot(1, 2, 1)
        plt.bar(range(len(self.region_importance)), self.region_importance)
        plt.xlabel('Region Index')
        plt.ylabel('Importance Score')
        plt.title('Region Importance Scores')
        
        # Highlight selected regions
        for idx in self.selected_indices:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)
        
        # Plot only selected regions
        plt.subplot(1, 2, 2)
        selected_importance = self.region_importance[self.selected_indices]
        plt.bar(range(len(selected_importance)), selected_importance)
        plt.xlabel('Selected Region Index')
        plt.ylabel('Importance Score')
        plt.title(f'Top {self.n_selected_regions} Selected Regions')
        plt.xticks(range(len(selected_importance)), self.selected_indices, rotation=45)
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_subjects = 360
    n_regions = 165
    n_features = 5
    n_selected_regions = 15

    with open('./data/gmwm3D'+'.pkl','rb') as f:
      data=pickle.load(f)
   
    gmwm3D=data['gmwm3D']
    gmwmDF=data['df']
    gmwmDim=data['gmwmDimensions']
    uniqueHeaders=data['uniqueHeaders']
    siteList=data['siteList']

    
    # Create synthetic data with some structure
    # Some regions will have stronger signals than others
    X = gmwm3D
    
    # Make some regions more important (stronger signal)
    #important_regions = np.random.choice(n_regions, 30, replace=False)
    #X[:, important_regions, :] *= 3.0
    
    print(f"Input data shape: {X.shape}")
    
    # Initialize and train the region selector
    selector = RegionSelector(
        n_regions=n_regions,
        n_features=n_features,
        n_selected_regions=n_selected_regions
    )
    
    # Train the model
    train_losses, val_losses = selector.train(
        X,
        epochs=10,
        learning_rate=0.001,
        batch_size=32,
        regularization_weight=0.01,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Plot training history
    selector.plot_training_history(train_losses, val_losses)
    
    # Plot region importance
    selector.plot_region_importance()
    
    # Transform data to selected regions
    X_reduced = selector.transform(X)
    print(f"\nReduced data shape: {X_reduced.shape}")

    
    # Get bottleneck representation
    bottleneck = selector.get_reduced_representation(X)
    print(f"Bottleneck representation shape: {bottleneck.shape}")
    
    # Print region importance scores for top regions
    print("\nRegion Importance Scores (top 15):")
    for i, idx in enumerate(selector.selected_indices):
        print(f"Region {idx}: {selector.region_importance[idx]:.4f}")

    # save 
    data={
      'gmwm3D_reduced':X_reduced,
      'df':gmwmDF,
      'gmwmDimensions':gmwmDim,
      'uniqueHeaders':uniqueHeaders,
      'siteList':siteList
      }
    outName='./data/reducedGMWM'
    with open(outName+'.pkl','wb') as f:
      pickle.dump(data,f)

if __name__ == "__main__":
    main()