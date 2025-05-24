# Suggested fixes for train_clique.py issues

# 1. Fix data splitting with proper shuffling
def fix_data_split(all_examples):
    import random
    # Shuffle before splitting to avoid temporal bias
    shuffled_examples = all_examples.copy()
    random.shuffle(shuffled_examples)
    
    train_size = int(0.9 * len(shuffled_examples))
    train_examples = shuffled_examples[:train_size]
    val_examples = shuffled_examples[train_size:]
    return train_examples, val_examples

# 2. Fix early stopping to always return best model
def fix_early_stopping_logic():
    """
    Current issue: best model only restored during early stopping
    Fix: Always restore best model at the end
    """
    # After training loop ends, add:
    # if best_model_state is not None:
    #     net.load_state_dict(best_model_state)
    #     print(f"Restored best model with loss: {best_loss:.6f}")

# 3. Fix device consistency
def fix_device_handling(net, device):
    """
    Keep model on same device for training and validation
    Only fall back to CPU if absolutely necessary
    """
    try:
        net.to(device)
        # Clear GPU cache if switching to GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return device
    except Exception as e:
        print(f"Failed to move to {device}, falling back to CPU: {e}")
        fallback_device = torch.device("cpu")
        net.to(fallback_device)
        return fallback_device

# 4. Add proper validation metrics calculation
def calculate_validation_metrics(net, val_examples, device):
    """
    Improved validation with better error handling
    """
    net.eval()
    total_policy_loss = 0
    total_value_loss = 0
    num_valid_examples = 0
    
    with torch.no_grad():
        for i, example in enumerate(val_examples):
            try:
                # Convert and validate data
                data = CliqueGameData([example])[0]
                data = data.to(device)
                
                # Forward pass
                policy_output, value_output = net(data.edge_index, data.edge_attr)
                
                # Numerical stability
                policy_output = torch.clamp(policy_output, min=1e-8)
                policy_output = policy_output / policy_output.sum()
                
                # Calculate losses
                policy_loss = -torch.sum(data.policy * torch.log(policy_output))
                value_loss = F.mse_loss(value_output.squeeze(), data.value)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_valid_examples += 1
                
            except Exception as e:
                print(f"Skipping invalid example {i}: {e}")
                continue
    
    if num_valid_examples > 0:
        return total_policy_loss / num_valid_examples, total_value_loss / num_valid_examples
    else:
        return 0.0, 0.0

# 5. Complete training function with fixes
def improved_train_network(all_examples, iteration, num_vertices, clique_size, model_dir, args):
    """
    Improved training function with all fixes applied
    """
    # Fix 1: Proper data shuffling
    train_examples, val_examples = fix_data_split(all_examples)
    
    # Initialize network
    net = CliqueGNN(num_vertices, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    
    # Fix 3: Consistent device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = fix_device_handling(net, device)
    
    # Training loop with fixes
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        try:
            epoch_loss = net.train_network(train_examples, 0, 
                                         args.epochs, device=device, args=args)
            
            # Fix 2: Always track best model
            if epoch_loss < best_loss - 0.001:  # min_delta
                best_loss = epoch_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(net.state_dict())
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= 5:  # patience
                break
                
        except Exception as e:
            print(f"Training error in epoch {epoch}: {e}")
            break
    
    # Fix 2: Always restore best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        print(f"Restored best model with loss: {best_loss:.6f}")
    
    # Fix 4: Improved validation
    avg_policy_loss, avg_value_loss = calculate_validation_metrics(net, val_examples, device)
    
    return avg_policy_loss, avg_value_loss 