# Transfer Learning Attempt: n=9,k=4 → n=13,k=4

## Date: August 31, 2025

## Objective
Transfer learned weights from a trained AlphaZero model (n=9, k=4) to initialize training for a larger graph (n=13, k=4), with the goal of accelerating convergence and preserving learned strategic knowledge.

## Background

### Source Model
- **Training**: n=9, k=4 (36 edges)
- **Status**: Trained for 13 iterations
- **Performance**: 
  - Iteration 9: 100% win rate vs initial model (breakthrough)
  - Iterations 10-13: Oscillating between 0% and 52% win rate (continued learning/instability)
- **Architecture**: GNN with 64 hidden dimensions, 3 layers

### Target Configuration
- **Goal**: n=13, k=4 (78 edges)
- **Hypothesis**: GNNs are naturally size-agnostic, processing graphs dynamically

## Implementation Attempts

### Attempt 1: Complex Weight Mapping (Failed)
**File**: `transfer_weights.py`

**Approach**:
- Flatten parameter trees and map by name
- Attempt to handle different shapes by padding/adaptation
- Reconstruct parameter tree structure

**Issues**:
- JAX parameter tree reconstruction failed
- String key corruption: `["['params']"]` instead of proper nested dict
- Parameters became inaccessible to model

**Result**: ❌ Model couldn't access transferred parameters

### Attempt 2: Direct Parameter Copy (Partially Successful)
**File**: `transfer_weights_fixed.py`

**Approach**:
- Direct copy of entire parameter dictionary
- Rely on GNN's size-agnostic nature
- No modification of weights

**Success**:
- ✅ Weights transferred (65 parameters)
- ✅ Model loads without errors
- ✅ Forward pass executes

**Issues**:
- Shape mismatch during evaluation (169 vs 78 expected edges)
- JAX JIT compilation extremely slow on CPU
- GPU memory conflicts with ongoing training

**Result**: ⚠️ Technical success but couldn't verify performance

## Key Findings

### 1. GNN Architecture Considerations

**Theory**: GNNs should be size-agnostic because:
- Message passing operates on local neighborhoods
- Node features start as zeros (no size dependency)
- Edge processing is dynamic

**Reality**: 
- Output dimensions appear coupled to training graph size
- Policy head may have implicit size dependencies
- Architecture not as portable as expected

### 2. Technical Challenges

**JAX/Flax Issues**:
- Complex nested parameter structure difficult to manipulate
- JIT compilation overhead prohibitive on CPU (~2+ minutes)
- Memory pre-allocation conflicts when GPU in use

**Evaluation Difficulties**:
- Couldn't complete evaluation games due to timeouts
- Shape mismatches prevented direct comparison
- No concrete win rate data obtained

### 3. Performance Results

**Attempted Evaluations**:
1. 51 games with 10 MCTS sims - Timed out
2. 10 games with 5 MCTS sims - Timed out  
3. 20 games without MCTS - Shape mismatch error
4. 2 demonstration games - Compilation timeout

**Actual Performance**: ❓ **Unknown** - couldn't complete evaluation

## Lessons Learned

### What Worked
1. **Parameter Transfer**: Successfully copied weights between models
2. **Model Loading**: Transferred model initializes and loads
3. **Concept Validation**: GNN weights can technically work across sizes

### What Didn't Work
1. **Performance Verification**: Couldn't demonstrate actual benefit
2. **Architecture Portability**: Output dimensions not truly size-agnostic
3. **Practical Evaluation**: JAX overhead made testing impractical

### Root Causes
1. **Architecture Design**: Policy head likely has hardcoded output size
2. **JAX Compilation**: CPU compilation time makes iteration difficult
3. **Resource Conflicts**: GPU memory competition with ongoing training

## Recommendations

### For Future Transfer Learning Attempts

1. **Architecture Modifications**:
   - Design truly size-agnostic policy head
   - Use dynamic output sizing based on input
   - Test portability during model design phase

2. **Implementation Approach**:
   - Use PyTorch for easier debugging
   - Implement gradual transfer (some layers, not all)
   - Consider fine-tuning specific components

3. **Evaluation Strategy**:
   - Reserve dedicated GPU for evaluation
   - Use pre-compiled models to avoid JIT overhead
   - Start with smaller size jumps (n=9 → n=10 → n=11)

### Immediate Next Steps

Given the challenges encountered, recommended approach:

1. **Option A**: Start fresh training for n=13,k=4
   - No transfer learning complications
   - Clean baseline for comparison
   - Proven training pipeline

2. **Option B**: Fix architecture for true size-agnosticism
   - Modify policy head to dynamically size output
   - Requires significant code changes
   - Higher risk but better long-term solution

## Code Artifacts

### Successfully Created
- `transfer_weights.py` - Initial attempt (flawed)
- `transfer_weights_fixed.py` - Working transfer script
- `evaluate_transfer.py` - Full evaluation with MCTS
- `evaluate_transfer_simple.py` - Simplified evaluation without MCTS
- `test_transfer.py` - Basic functionality test
- `quick_test.py` - Minimal demonstration attempt

### Checkpoints Generated
- `checkpoint_n13k4_transferred.pkl` - Corrupted structure (deleted)
- `checkpoint_n13k4_transferred_v2.pkl` - Working transfer checkpoint

## Conclusion

While the transfer learning attempt demonstrated that GNN weights can technically be transferred between different graph sizes, we could not verify actual performance benefits due to architectural limitations and technical challenges with JAX. The concept remains sound but requires better architecture design and implementation approach for practical success.

**Status**: ⚠️ **Partially Successful** - Transfer works technically but performance benefit unverified

**Recommendation**: Proceed with fresh training for n=13,k=4 rather than debugging transfer learning further.