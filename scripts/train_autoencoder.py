from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_io import load_activation_data


class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, latent_dim, top_k_percentage=0.01):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.top_k_percentage = top_k_percentage
        
    def batch_top_k(self, x):
        batch_size = x.shape[0]
        # Calculate k based on batch size and desired percentage
        k = max(1, int(batch_size * self.top_k_percentage))
        
        # Find top-k values for each neuron across the batch
        values, indices = torch.topk(x, k, dim=0)
        
        # Create a mask of zeros
        mask = torch.zeros_like(x)
        
        # Fill in the top-k values
        for i in range(k):
            mask[indices[i], torch.arange(x.shape[1])] = 1
            
        # Apply mask to activations
        return x * mask
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Apply BatchTopK sparsity
        sparse_encoded = self.batch_top_k(encoded)
        
        # Decode
        decoded = self.decoder(sparse_encoded)
        
        return decoded, sparse_encoded
    
def train_step(model, optimizer, batch):
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed, encoded = model(batch)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, batch)
    
    # No need for L1 loss since sparsity is enforced by BatchTopK
    
    # Backward pass
    recon_loss.backward()
    optimizer.step()
    
    return recon_loss.item()

def train_autoencoder(
    input_dim: int = 3072,  # GPT-Neo-125M MLP output dimension (4x hidden_dim)
    latent_dim: int = 6144,  # 2x input_dim for better feature learning
    batch_size: int = 128,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,  # Reduced default learning rate
    top_k_percentage: float = 0.01,
    save_dir: str = "models"
):
    # Load data
    data = load_activation_data()
    activations = data["activations"]
    
    # Create dataset and dataloader
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = BatchTopKSAE(input_dim, latent_dim, top_k_percentage)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose='True')
    
    # Training loop
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10  # Early stopping patience
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting training...")
    print(f"Model architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Top-k percentage: {top_k_percentage}")
    print(f"Training parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        model.train()
        
        for batch_idx, (batch,) in enumerate(dataloader):
            loss = train_step(model, optimizer, batch)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            epoch_losses.append(loss)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss:.6f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(avg_epoch_loss)
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'top_k_percentage': top_k_percentage
                }
            }, str(save_path / 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(str(save_path / 'training_curve.png'))
    plt.close()
    
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Model and training curve saved to {save_dir}")
    
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a BatchTopK Sparse Autoencoder')
    parser.add_argument('--input-dim', type=int, default=3072,
                      help='Input dimension (default: 3072)')
    parser.add_argument('--latent-dim', type=int, default=6144,
                      help='Latent dimension (default: 6144)')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of epochs (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate (default: 1e-4)')
    parser.add_argument('--top-k-percentage', type=float, default=0.01,
                      help='Top-k percentage for sparsity (default: 0.01)')
    parser.add_argument('--save-dir', type=str, default='models',
                      help='Directory to save models and plots (default: models)')
    
    args = parser.parse_args()
    
    model = train_autoencoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        top_k_percentage=args.top_k_percentage,
        save_dir=args.save_dir
    )
    return model


if __name__ == "__main__":
    main()
    