import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoForCausalLM

from ..src.data_io import save_activation_data


def setup_model() -> Tuple[GPTNeoForCausalLM, AutoTokenizer, str]:
    """Initialize the model and tokenizer."""
    model_name = "EleutherAI/gpt-neo-125m"
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)   # type: ignore
    
    # Set up tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device # type: ignore


def collect_activations(
    model: GPTNeoForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    num_examples: int,  # Increased for better feature learning
    sequence_length: int
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Collect GELU activations from layer 6 for individual token positions."""
    activations = []
    tokens = []
    texts = []
    
    def hook_fn(module, input, output):
        # output shape: [batch_size, sequence_length, hidden_dim]
        # Randomly select one token position per sequence
        batch_size = output.shape[0]
        for i in range(batch_size):
            pos = torch.randint(0, output.shape[1], (1,)).item()
            # Store activation for just that token
            activations.append(output[i, pos].detach().cpu())

    # Register hook on layer 6 GELU
    gelu = model.transformer.h[6].mlp.act
    hook = gelu.register_forward_hook(hook_fn)
    
    # Load and process examples
    dataset = load_dataset('ag_news', split=f'train[:{num_examples}]')
    
    for i, example in enumerate(dataset):
        text = example['text']
        texts.append(text)

        inputs = tokenizer( # type: ignore
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=sequence_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Store the actual tokens for analysis
        tokens.append(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])) # type: ignore
        
        if i % 100 == 0:
            print(f"Processed {i} examples")
    
    hook.remove()
    
    # Stack all activations into a single tensor
    activations_tensor = torch.stack(activations)
    print(f"Collected activation tensor shape: {activations_tensor.shape}")
    
    return activations_tensor, tokens, texts # type: ignore

# TODO: unused
def generate_sample_text(
    model: GPTNeoForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt: str = "The quick brown"
) -> Tuple[str, float]:
    """Generate sample text and measure generation time."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device) # type: ignore
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=20)   # type: ignore
    generation_time = time.time() - start_time
    
    result = tokenizer.decode(outputs[0]) # type: ignore
    return result, generation_time


def main():
    # Setup
    model, tokenizer, device = setup_model()
    print(f"Using device: {device}")
    
    # Collect activations
    print("\nCollecting activations...")
    activations, tokens, texts = collect_activations(model, tokenizer, device, num_examples=10, sequence_length=128)

    # Save activations
    save_activation_data(
        activations=activations,
        tokens=tokens,
        texts=texts,
        save_dir="data/activations",
        batch_name="ag_news_batch_test"
    )

    # print(f"Activations: {activations}")
    # print(f"Tokens: {tokens}")
    # print(f"Texts: {texts}")
    
    # print("\nCollected activation shapes:")
    # for i, act in enumerate(activations):
    #     print(f"Example {i}: {act.shape}")
    

if __name__ == "__main__":
    main()