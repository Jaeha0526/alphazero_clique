#!/usr/bin/env python
import torch
from alpha_net_clique import CliqueGNN
from clique_board import CliqueBoard
from encoder_decoder_clique import prepare_state_for_network

def test_model():
    # Create a simple board for testing
    board = CliqueBoard(num_vertices=6, k=3)
    
    # Create GNN model
    model = CliqueGNN(num_vertices=6)
    
    # Set model to eval mode
    model.eval()
    
    # Prepare board state for the network
    state_dict = prepare_state_for_network(board)
    edge_index = state_dict['edge_index']
    edge_attr = state_dict['edge_attr']
    
    # Print shapes for debugging
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    
    # Run model inference
    with torch.no_grad():
        policy, value = model(edge_index, edge_attr)
    
    # Print results
    print(f"Policy shape: {policy.shape}")
    print(f"Value: {value.item()}")
    
    print("Model test successful!")

if __name__ == "__main__":
    try:
        test_model()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}") 