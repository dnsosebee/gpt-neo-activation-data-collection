import json
from datetime import datetime
from pathlib import Path

import torch

from scripts.train_autoencoder import BatchTopKSAE
from src.data_io import load_activation_data


def load_trained_model(model_path):
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    model = BatchTopKSAE(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        top_k_percentage=config['top_k_percentage']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config, checkpoint['epoch']

def extract_features(model, data, batch_size=128):
    """Extract features from the model in batches to avoid memory issues"""
    features = []
    n_samples = len(data)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            _, encoded = model(batch)
            features.append(encoded)
    
    return torch.cat(features, dim=0)

def main():
    # Create output directory
    output_dir = Path("data/sae_features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print("Loading model and data...")
    model, config, trained_epoch = load_trained_model('models/best_model.pt')
    data = load_activation_data()
    activations = data["activations"]
    metadata = data.get("metadata", {})
    
    # Extract features
    print("Extracting SAE features...")
    features = extract_features(model, activations)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save features
    features_path = output_dir / f"sae_features_{timestamp}.pt"
    torch.save(features, features_path)
    
    # Save metadata
    metadata_dict = {
        "original_data_metadata": metadata,
        "sae_config": config,
        "trained_epoch": trained_epoch,
        "extraction_timestamp": timestamp,
        "num_samples": len(features),
        "feature_dim": features.shape[1],
        "sparsity": (features == 0).float().mean().item(),
        "mean_activation": features.abs().mean().item(),
        "features_file": features_path.name
    }
    
    metadata_path = output_dir / f"sae_features_{timestamp}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Features shape: {features.shape}")
    print(f"Features saved to: {features_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nFeature statistics:")
    print(f"  Average sparsity: {metadata_dict['sparsity']:.3f}")
    print(f"  Average activation: {metadata_dict['mean_activation']:.3f}")

if __name__ == "__main__":
    main() 