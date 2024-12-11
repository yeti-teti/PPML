import os
import torch
import argparse
from fl_server import FederatedServer
from fl_client import FederatedClient

def train_federated(num_clients=3, num_rounds=5, batch_size=4, block_size=1024):
    """Main training function"""
    
    # Setup
    os.makedirs('out-fl', exist_ok=True)
    torch.manual_seed(1337)  # for reproducibility
    
    # Initialize server
    print("\nInitializing server...")
    server = FederatedServer()
    
    # Initialize clients
    print("\nInitializing clients...")
    clients = [
        FederatedClient(
            client_id=i,
            batch_size=batch_size,
            block_size=block_size
        ) for i in range(num_clients)
    ]
    
    # Training loop
    best_val_loss = float('inf')
    
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
        
        # Aggregate models
        print("\nAggregating models...")
        client_models = [client.get_model_state() for client in clients]
        server.aggregate_models(client_models)
        
        # Track progress
        avg_val_loss = sum(client_losses) / len(client_losses)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint
        server.save_model(round_num)
        
        # Round summary
        print(f"\nRound {round_num + 1} Summary:")
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
    
    print("\nTraining completed!")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--num_clients', type=int, default=3,
                      help='number of clients (default: 3)')
    parser.add_argument('--num_rounds', type=int, default=5,
                      help='number of training rounds (default: 5)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='training batch size (default: 4)')
    parser.add_argument('--block_size', type=int, default=1024,
                      help='sequence block size (default: 1024)')
    args = parser.parse_args()
    
    print("Starting Federated Learning Training")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    
    train_federated(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        block_size=args.block_size
    )

if __name__ == "__main__":
    main()