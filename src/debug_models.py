#!/usr/bin/env python

import torch
import os
import glob

# Get project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
playable_model_dir = os.path.join(project_root, 'playable_models')

print(f"Searching for models in: {playable_model_dir}")

model_files = glob.glob(os.path.join(playable_model_dir, "*.pth.tar"))
print(f"Found {len(model_files)} model files:")

for model_path in model_files:
    filename = os.path.basename(model_path)
    print(f"\n=== {filename} ===")
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"Keys: {list(checkpoint.keys())}")
        
        for key in ['num_vertices', 'clique_size', 'hidden_dim', 'num_layers']:
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")
            else:
                print(f"  {key}: MISSING")
                
        # Check for attention layers in state_dict
        if 'state_dict' in checkpoint:
            state_keys = list(checkpoint['state_dict'].keys())
            has_attention = any('attention' in key for key in state_keys)
            print(f"  Has attention layers: {has_attention}")
            
            # Show a few sample keys
            print(f"  Sample state_dict keys: {state_keys[:5]}...")
                
    except Exception as e:
        print(f"  ERROR: {e}") 