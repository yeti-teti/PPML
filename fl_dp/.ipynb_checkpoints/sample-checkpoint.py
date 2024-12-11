import os
import torch
import argparse
import tiktoken
from model import GPTConfig, GPT
from privacy_utils import add_noise_to_logits

def load_model(model_path):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cuda')
    
    # Create model with saved configuration
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    
    return model

def generate_sample(model, prompt, max_new_tokens=100, temperature=0.8, 
                   top_k=40, noise_scale=0.1, device='cuda'):
    """Generate text with privacy-preserving noise"""
    # Prepare model
    model = model.to(device)
    model.eval()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Generate tokens
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Add noise to final logits
        if noise_scale > 0:
            output_ids = add_noise_to_logits(output_ids, noise_scale)
    
    # Decode and return text
    output_text = enc.decode(output_ids[0].tolist())
    return output_text

def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--model_path', default='out-fl/best_model.pt')
    parser.add_argument('--num_samples', type=