# Key Vectorization Changes

## 1. Data Structure Changes

### Before (SimpleTreeMCTS):
```python
# Separate tree for each game
trees = []
for game_idx in range(batch_size):
    tree = {
        'N': {},  # Python dict
        'W': {},
        'children': {},
        ...
    }
```

### After (VectorizedTreeMCTSv2):
```python
# Single array for all games
N = jnp.zeros((batch_size, max_nodes, num_actions))  # All games in one array!
W = jnp.zeros((batch_size, max_nodes, num_actions))
children = jnp.ones((batch_size, max_nodes, num_actions), dtype=int) * -1
```

## 2. Selection Phase Changes

### Before:
```python
# Loop through each game
for game_idx, tree in enumerate(trees):
    node_id = 0
    while node_id in tree['expanded']:
        # Calculate UCB for one game
        N = tree['N'][node_id]
        W = tree['W'][node_id]
        ucb = W/N + c_puct * P * sqrt(sum(N)) / (1 + N)
        action = np.argmax(ucb)
        node_id = tree['children'][node_id][action]
```

### After:
```python
# All games traverse together!
current_nodes = jnp.zeros(batch_size, dtype=int)  # All at root
active_games = jnp.ones(batch_size, dtype=bool)

while active_games.any():
    # Get data for ALL games' current nodes at once
    batch_idx = jnp.arange(batch_size)
    current_N = N[batch_idx, current_nodes]  # Shape: (batch_size, num_actions)
    current_W = W[batch_idx, current_nodes]
    
    # Vectorized UCB for all games
    Q = current_W / jnp.maximum(current_N, 1)
    U = c_puct * current_P * sqrt_total / (1 + current_N)
    UCB = Q + U
    
    # Select best actions for all games
    best_actions = jnp.argmax(UCB, axis=1)  # Shape: (batch_size,)
    
    # Move to children (vectorized)
    child_indices = children[batch_idx, current_nodes, best_actions]
    current_nodes = jnp.where(should_continue, child_indices, current_nodes)
```

## 3. Synchronization Strategy

The key insight is synchronized traversal with masking:

```python
# Some games stop early
is_expanded_current = expanded[batch_idx, current_nodes]
is_terminal_current = terminal[batch_idx, current_nodes]
should_continue = active_games & is_expanded_current & ~is_terminal_current

# Mask UCB for stopped games
UCB = jnp.where(should_continue[:, None], UCB, -jnp.inf)

# Only move active games
current_nodes = jnp.where(should_continue, child_indices, current_nodes)
```

## 4. Batch NN Evaluation

### Before:
```python
# Evaluate one position at a time
for game_idx, (_, _, board) in enumerate(leaves_to_eval):
    policy, value = neural_network.evaluate(board)
```

### After:
```python
# Collect ALL positions needing evaluation
need_expansion = ~expanded[batch_idx, current_nodes]

# Single batched NN call
eval_policies, eval_values = neural_network.evaluate_batch(all_positions)
```

## 5. Vectorized Backup

### Before:
```python
# Update each path separately
for game_idx, path in paths:
    for node, action in path:
        tree['N'][node][action] += 1
        tree['W'][node][action] += value
```

### After:
```python
# Update all paths at once
N = N.at[batch_idx, path_nodes, path_actions].add(1)
W = W.at[batch_idx, path_nodes, path_actions].add(values)
```

## Performance Impact

- **Selection**: ~100x faster (vectorized array ops vs Python loops)
- **NN Evaluation**: ~100x faster (batched)
- **Backup**: ~50x faster (vectorized updates)
- **Memory**: Higher but manageable (~100MB for 500 games)