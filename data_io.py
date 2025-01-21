import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def save_activation_data(
    activations: torch.Tensor,
    tokens: List[str],
    texts: List[str],
    save_dir: str = "data/activations",
    batch_name: str = "batch_1"
) -> None:
    """Save activation data and metadata to files."""
    # Create directory if it doesn't exist
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save activations tensor
    activation_path = save_dir_path / f"{batch_name}_activations.pt"
    torch.save(activations, activation_path)
    
    # Save metadata
    metadata = {
        "activation_shape": list(activations.shape),
        "num_samples": len(tokens),
        "tokens": tokens,
        "texts": texts,
    }
    
    metadata_path = save_dir_path / f"{batch_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved activation data to {save_dir}")
    print(f"Activation shape: {activations.shape}")
    print(f"Number of samples: {len(tokens)}")


def load_activation_data(
    load_dir: str = "data/activations",
    batch_name: str = "batch_1"
) -> Dict[str, Any]:
    """Load activation data and metadata from files."""
    load_dir_path = Path(load_dir)
    
    # Load activations
    activation_path = load_dir_path / f"{batch_name}_activations.pt"
    activations = torch.load(activation_path)
    
    # Load metadata
    metadata_path = load_dir_path / f"{batch_name}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {
        "activations": activations,
        "tokens": metadata["tokens"],
        "texts": metadata["texts"],
        "metadata": metadata
    } 