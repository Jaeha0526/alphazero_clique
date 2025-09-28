#!/usr/bin/env python
"""
Run JAX evaluation in separate subprocesses to enable CPU parallelization.
This avoids JAX's multiprocessing issues by running each worker as a separate Python process.
"""

import subprocess
import pickle
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional
import time


def evaluate_worker_subprocess(
    worker_id: int,
    model1_path: str,
    model2_path: str,
    num_games: int,
    config: dict,
    output_file: str
):
    """
    Run evaluation in a subprocess to avoid JAX threading issues.
    
    This function is called as a separate Python process.
    """
    # This code runs in the subprocess
    code = f"""
import pickle
import json
from pathlib import Path

# Import evaluation code
from standalone_evaluation import load_model
from evaluation_jax_parallel import evaluate_models_parallel

# Load models
model1 = load_model('{model1_path}')
model2 = load_model('{model2_path}')

# Run evaluation
results = evaluate_models_parallel(
    model1=model1,
    model2=model2,
    num_games={num_games},
    num_vertices={config['num_vertices']},
    k={config['k']},
    mcts_sims={config['mcts_sims']},
    c_puct={config.get('c_puct', 3.0)},
    temperature=0.0,
    game_mode='{config.get('game_mode', 'symmetric')}',
    python_eval={config.get('python_eval', True)}
)

# Add worker id to results
results['worker_id'] = {worker_id}

# Save results
with open('{output_file}', 'wb') as f:
    pickle.dump(results, f)

print(f"Worker {worker_id} completed: {{results['model1_wins']}} vs {{results['model2_wins']}} ({{results['draws']}} draws)")
"""
    
    # Run as subprocess
    result = subprocess.run(
        ['python', '-c', code],
        capture_output=True,
        text=True,
        cwd='jax_full_src'
    )
    
    if result.returncode != 0:
        print(f"Worker {worker_id} failed: {result.stderr}")
        return None
    
    return output_file


def evaluate_models_subprocess_parallel(
    model1_path: str,
    model2_path: str,
    num_games: int = 40,
    num_cpus: int = 4,
    config: Optional[Dict] = None
) -> Dict:
    """
    Evaluate models using subprocess-based parallelization.
    
    Each worker runs in a completely separate Python process,
    avoiding JAX's threading issues.
    
    Args:
        model1_path: Path to first model checkpoint
        model2_path: Path to second model checkpoint
        num_games: Total games to play
        num_cpus: Number of parallel workers
        config: Evaluation configuration
        
    Returns:
        Aggregated evaluation results
    """
    if config is None:
        config = {
            'num_vertices': 6,
            'k': 3,
            'mcts_sims': 30,
            'game_mode': 'symmetric',
            'python_eval': True
        }
    
    print(f"\nSubprocess Parallel Evaluation")
    print(f"  Models: {Path(model1_path).name} vs {Path(model2_path).name}")
    print(f"  Games: {num_games} using {num_cpus} workers")
    print(f"  Config: n={config['num_vertices']}, k={config['k']}, sims={config['mcts_sims']}")
    
    start_time = time.time()
    
    # Calculate games per worker
    games_per_worker = num_games // num_cpus
    remainder = num_games % num_cpus
    
    # Create temporary directory for results
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Start workers as subprocesses
        processes = []
        result_files = []
        
        for i in range(num_cpus):
            worker_games = games_per_worker + (1 if i < remainder else 0)
            
            if worker_games > 0:
                # Output file for this worker
                output_file = tmpdir / f'worker_{i}_results.pkl'
                result_files.append(output_file)
                
                # Create worker script
                worker_script = tmpdir / f'worker_{i}.py'
                with open(worker_script, 'w') as f:
                    f.write(f"""
# IMPORTANT: Set environment variables BEFORE importing anything
import os
import sys

# Force CPU-only mode with all available flags
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_ENABLE_X64'] = 'False'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_DISABLE_JIT'] = 'False'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

# Suppress all warnings before importing JAX
import warnings
warnings.filterwarnings('ignore')

# Disable JAX GPU plugin discovery completely
os.environ['JAX_PLATFORMS'] = 'cpu'

import pickle
sys.path.append('jax_full_src')

# Try to import JAX and force CPU backend
try:
    import jax
    # Force CPU backend explicitly
    jax.config.update('jax_platform_name', 'cpu')
except Exception as e:
    print(f"Worker {i}: Warning during JAX import: {{e}}", file=sys.stderr)
    pass

from standalone_evaluation import load_model
from evaluation_jax_parallel import evaluate_models_parallel

# Load models
print(f"Worker {i}: Loading models...")
model1 = load_model('{model1_path}')
model2 = load_model('{model2_path}')

print(f"Worker {i}: Evaluating {worker_games} games...")
results = evaluate_models_parallel(
    model1=model1,
    model2=model2,
    num_games={worker_games},
    num_vertices={config['num_vertices']},
    k={config['k']},
    mcts_sims={config['mcts_sims']},
    c_puct={config.get('c_puct', 3.0)},
    temperature=0.0,
    game_mode='{config.get('game_mode', 'symmetric')}',
    python_eval={config.get('python_eval', True)}
)

results['worker_id'] = {i}

with open('{output_file}', 'wb') as f:
    pickle.dump(results, f)

print(f"Worker {i}: Completed {{results['model1_wins']}}-{{results['model2_wins']}}-{{results['draws']}}")
""")
                
                # Start subprocess with environment variables
                print(f"  Starting worker {i}: {worker_games} games")
                env = os.environ.copy()
                # Force CPU-only execution
                env['JAX_PLATFORMS'] = 'cpu'
                env['CUDA_VISIBLE_DEVICES'] = ''  # Empty string disables CUDA properly
                env['OMP_NUM_THREADS'] = '1'  # Limit threads per process
                env['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'  # Force single CPU device
                # Disable all GPU-related libraries
                env['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
                env['JAX_ENABLE_X64'] = 'False'
                env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
                # Run with explicit python flags to reduce memory usage
                proc = subprocess.Popen(
                    ['python', '-u', str(worker_script)],  # -u for unbuffered output
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                processes.append((i, proc))
        
        # Wait for all workers to complete
        print("\n  Waiting for workers...")
        successful_workers = []
        failed_workers = []

        for i, proc in processes:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                # Filter out known JAX CUDA warnings that are not fatal
                stderr_lines = stderr.split('\n')
                fatal_errors = [line for line in stderr_lines
                               if line and 'CUDA_ERROR' not in line
                               and 'cuda_versions' not in line
                               and 'jax_plugins' not in line
                               and 'Jax plugin configuration error' not in line
                               and 'pthread_create' not in line
                               and 'tf_XLAEigen' not in line
                               and 'tf_xla-cpu-llvm' not in line]
                if fatal_errors:
                    print(f"  Worker {i} failed with errors")
                    failed_workers.append(i)
                else:
                    # Likely just JAX initialization warnings, check if result file exists
                    result_file = tmpdir / f'worker_{i}_results.pkl'
                    if result_file.exists():
                        print(f"  Worker {i}: Completed (with warnings)")
                        successful_workers.append(i)
                    else:
                        print(f"  Worker {i}: Failed (no results)")
                        failed_workers.append(i)
            else:
                # Print only the completion line from stdout
                for line in stdout.split('\n'):
                    if 'Completed' in line:
                        print(f"  {line.strip()}")
                successful_workers.append(i)

        if failed_workers:
            print(f"\n  ⚠️  {len(failed_workers)} workers failed, using results from {len(successful_workers)} successful workers")
        
        # Aggregate results from successful workers only
        total_model1_wins = 0
        total_model2_wins = 0
        total_draws = 0
        total_games_played = 0

        for result_file in result_files:
            if result_file.exists():
                try:
                    with open(result_file, 'rb') as f:
                        results = pickle.load(f)
                        total_model1_wins += results['model1_wins']
                        total_model2_wins += results['model2_wins']
                        total_draws += results['draws']
                        # Calculate total games from wins and draws
                        games_in_batch = results['model1_wins'] + results['model2_wins'] + results['draws']
                        total_games_played += games_in_batch
                except Exception as e:
                    print(f"  Warning: Could not load {result_file.name}: {e}")

        if total_games_played == 0:
            print(f"\n  ⚠️  ERROR: No games completed successfully!")
            return {
                'model1_wins': 0,
                'model2_wins': 0,
                'draws': 0,
                'total_games': 0,
                'model1_win_rate': 0.5,
                'model2_win_rate': 0.5,
                'draw_rate': 0.0,
                'eval_time': time.time() - start_time,
                'games_per_second': 0,
                'error': 'All workers failed'
            }
    
    # Calculate rates
    model1_win_rate = total_model1_wins / total_games_played if total_games_played > 0 else 0
    model2_win_rate = total_model2_wins / total_games_played if total_games_played > 0 else 0
    draw_rate = total_draws / total_games_played if total_games_played > 0 else 0
    
    eval_time = time.time() - start_time
    
    final_results = {
        'model1_wins': total_model1_wins,
        'model2_wins': total_model2_wins,
        'draws': total_draws,
        'total_games': total_games_played,
        'model1_win_rate': model1_win_rate,
        'model2_win_rate': model2_win_rate,
        'draw_rate': draw_rate,
        'eval_time': eval_time,
        'games_per_second': total_games_played / eval_time if eval_time > 0 else 0
    }
    
    print(f"\nResults ({total_games_played} games in {eval_time:.1f}s):")
    print(f"  Model1: {total_model1_wins} wins ({model1_win_rate:.1%})")
    print(f"  Model2: {total_model2_wins} wins ({model2_win_rate:.1%})")
    print(f"  Draws: {total_draws} ({draw_rate:.1%})")
    print(f"  Speed: {final_results['games_per_second']:.1f} games/sec")
    
    return final_results


if __name__ == "__main__":
    print("Subprocess-based Parallel Evaluation for JAX Models")
    print("="*60)
    print("This runs each worker in a separate Python process,")
    print("completely avoiding JAX's threading/forking issues.")
    print("="*60)
    
    # Test with existing models
    from pathlib import Path
    
    experiments = list(Path('experiments').glob('*/checkpoints/*.pkl'))
    if len(experiments) >= 2:
        model1 = str(experiments[0])
        model2 = str(experiments[1] if len(experiments) > 1 else experiments[0])
        
        print(f"\nTest evaluation:")
        print(f"  Model 1: {model1}")
        print(f"  Model 2: {model2}")
        
        results = evaluate_models_subprocess_parallel(
            model1_path=model1,
            model2_path=model2,
            num_games=8,
            num_cpus=2,
            config={
                'num_vertices': 4,
                'k': 3,
                'mcts_sims': 10,
                'game_mode': 'symmetric',
                'python_eval': True
            }
        )
    else:
        print("\nNot enough models found for test evaluation")