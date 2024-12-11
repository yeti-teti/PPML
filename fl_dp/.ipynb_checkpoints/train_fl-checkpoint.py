import os
import torch
import argparse
from fl_server import FederatedServer
from fl_client import FederatedClient
from privacy_utils import EarlyStopping

def train_federated(num_clients=3, 
                   num_rounds=5,
                   batch_size=4,
                   block_size=1024,
                   noise_multiplier=1.0,
                   max_grad_norm=1.0,
                   server_noise_scale=0.01):
    
    # Setup
    os.makedirs('out-fl', exist_ok=True)
    torch.manual_seed(1337)
    
    # Initialize server
    print("\nInitializing server...")
    server = FederatedServer(noise_scale=server_noise_scale)
    
    # Initialize clients
    print("\nInitializing clients...")
    clients = [
        FederatedClient(
            client_id=i,
            batch_size=batch_size,
            block_size=block_size,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        ) for i in range(num_clients)
    ]
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3)
    best_val_loss = float('inf')
    
    # Training loop
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
        # Train each client
        client_losses = []
        for client in clients:
            print(f"\nTraining Client {client.client_id}")
            
            # Update client with global model
            client.update_model(server.get_model_state())
            
            # Train and evaluate
            train_loss = client.train_epoch()
            val_loss = client.evaluate()
            client_losses.append(val_loss)
            
            print(f"Client {client.client_id} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Calculate average validation loss
        avg_val_loss = sum(client_losses) / len(client_losses)
        
        # Aggregate models
        print("\nAggregating models...")
        client_models = [client.get_model_state() for client in clients]
        server.aggregate_models(client_models)
        
        # Save best model if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            server.save_model(round_num)
            print(f"New best model saved! (Val Loss: {avg_val_loss:.6f})")
        
        # Check for early stopping
        if early_stopping(avg_val_loss):
            print("\nEarly stopping triggered!")
            break
        
        # Round summary
        print(f"\nRound {round_num + 1} Summary:")
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Privacy-Preserving Federated Learning')
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--noise_multiplier', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--server_noise_scale', type=float, default=0.01)
    args = parser.parse_args()
    
    print("Starting Privacy-Preserving Federated Learning")
    print(f"Configuration:")
    print(f"- Number of clients: {args.num_clients}")
    print(f"- Number of rounds: {args.num_rounds}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Privacy noise multiplier: {args.noise_multiplier}")
    print(f"- Max gradient norm: {args.max_grad_norm}")
    print(f"- Server noise scale: {args.server_noise_scale}")
    
    train_federated(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        block_size=args.block_size,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        server_noise_scale=args.server_noise_scale
    )

if __name__ == "__main__":
    main()