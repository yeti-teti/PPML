# server.py
import flwr as fl

def main():
    # Strategy: FedAvg is the default; you can customize it as needed
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # Fraction of clients used for training in each round
        fraction_evaluate=0.5,  # Fraction of clients used for evaluation in each round
        min_fit_clients=2,  # Minimum number of clients to participate in training
        min_evaluate_clients=2,  # Minimum number of clients to participate in evaluation
        min_available_clients=2,  # Minimum number of total clients
    )

    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
