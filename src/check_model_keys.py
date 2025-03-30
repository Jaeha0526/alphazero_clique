import torch
import os

# --- Configuration ---
# Assuming this script is in src/, playable_models is at the project root
project_root = os.path.dirname(os.path.abspath(__file__)) # src directory
project_root = os.path.dirname(project_root) # Project root directory

model_filename = "n6k3_exp2_clique_net.pth.tar"
# model_filename = "clique_net_iter2.pth.tar"
model_path = os.path.join(project_root, "playable_models", model_filename)
# --- End Configuration ---

print(f"Attempting to load model file: {model_path}")

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at the specified path.")
    print("Please ensure the file exists and the path is correct.")
else:
    try:
        # Load the checkpoint dictionary from the file
        # map_location='cpu' ensures it loads even if saved on GPU
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if isinstance(checkpoint, dict):
            print("\nSuccessfully loaded the file. It contains a dictionary.")
            print("Keys found in the dictionary:")
            # Print all keys
            for key in checkpoint.keys():
                print(f"- {key}")

            # Specifically check for the expected keys
            print("\nChecking for specific keys:")
            if 'num_vertices' in checkpoint:
                print(f"- Found 'num_vertices': {checkpoint['num_vertices']}")
            else:
                print("- Key 'num_vertices' is MISSING.")

            if 'clique_size' in checkpoint:
                print(f"- Found 'clique_size': {checkpoint['clique_size']}")
            else:
                print("- Key 'clique_size' is MISSING.")

            if 'state_dict' in checkpoint:
                print("- Found 'state_dict' (model weights).")
            else:
                print("- Key 'state_dict' is MISSING.")

        else:
            print("\nLoaded the file, but it does not contain a dictionary.")
            print(f"Type of loaded object: {type(checkpoint)}")

    except Exception as e:
        print(f"\nERROR: An error occurred while loading or reading the model file:")
        print(e)
        print("The file might be corrupted or not a valid PyTorch save file.") 