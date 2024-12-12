import os
import torch
import tiktoken
from model import GPTConfig, GPT

class ChatInterface:
    def __init__(self, model_path='out-medical/ckpt.pt', max_tokens=200, temperature=0.7):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = GPT(GPTConfig(**checkpoint['model_args']))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Load privacy budget info
        self.epsilon = checkpoint.get('epsilon', None)
        print(f"Model trained with privacy budget ε = {self.epsilon:.2f}")
    
    def generate_response(self, prompt):
        # Encode prompt
        input_ids = self.enc.encode(prompt)
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...]
        
        # Generate response
        with torch.no_grad():
            y = self.model.generate(
                x,
                self.max_tokens,
                temperature=self.temperature,
                top_k=40
            )
            response = self.enc.decode(y[0].tolist()[len(input_ids):])
        
        return response
    
    def chat(self):
        print("\nPrivately Trained GPT Chat Interface")
        print(f"Privacy guarantee: ε = {self.epsilon:.2f}")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            prompt = f"\nUser: {user_input}\nAssistant: "
            response = self.generate_response(prompt)
            
            print("\nAssistant:", response)
            print("\n" + "="*50)

if __name__ == "__main__":
    chat = ChatInterface()
    chat.chat()