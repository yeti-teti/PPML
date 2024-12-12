"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# Configuration
init_from = 'resume'
out_dir = 'out-medical'
start_prompts = [
    "\nPatient Record\n=================\n\nPatient ID: ",
    "\nGenerate a medical record for a patient with diabetes\n",
    "\nCreate a patient record for someone with hypertension\n",
    "\nShow me a medical record for an elderly patient\n",
    "\nGenerate a record for a young adult patient\n"
]
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float32'
compile = False

def main():
    # Setup
    torch.manual_seed(seed)
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # Load model
    print(f"Loading model from {out_dir}...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Initialize model with checkpoint args
    model = GPT(GPTConfig(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model_state_dict'])  # Changed from 'model' to 'model_state_dict'
    model.eval()
    model.to(device)
    
    if compile:
        model = torch.compile(model)
    
    # Generate samples
    print(f"\nGenerating samples from {len(start_prompts)} different prompts...\n")
    
    for i, prompt in enumerate(start_prompts, 1):
        print(f"Prompt {i}: {prompt}")
        print("="*50)
        
        # Encode and generate
        start_ids = encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        
        with torch.no_grad():
            with ctx:
                print("Generated output:")
                print("-"*20)
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print("-"*20 + "\n")

if __name__ == '__main__':
    main()