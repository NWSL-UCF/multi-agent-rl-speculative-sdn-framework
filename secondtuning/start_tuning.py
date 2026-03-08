import pandas as pd
import os
from itertools import product
import random
import json
import subprocess
import time
import shlex
import logging
import atexit
import signal
import zipfile
import asyncio
from pathlib import Path
import sys

# Add parent directory to path to import psc
sys.path.insert(0, str(Path(__file__).parent))
from psc import submit_job

# Constants
MAX_ITR = 7
CHECKPOINT_FILE = "checkpoint.json"
STATE_FILE = "state.json"

def load_init_config():
    """Load configuration from init.json"""
    init_file = Path(__file__).parent / "init.json"
    with open(init_file, 'r') as f:
        return json.load(f)

def setup_paths(exp_id, global_seed):
    """Setup experiment paths"""
    home_path = Path.home()
    exp_path = home_path / "data" / "raw" / exp_id / str(global_seed)
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path

def setup_logging(output_path):
    """Setup logging"""
    log_file = output_path / "tuning_log.txt"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True
    )
    return logging.getLogger()

def load_checkpoint(output_path):
    """Load checkpoint if exists"""
    checkpoint_file = output_path / CHECKPOINT_FILE
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint file found: {checkpoint_file}")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    return None

def save_checkpoint(output_path, state):
    """Save checkpoint"""
    checkpoint_file = output_path / CHECKPOINT_FILE
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")

def save_state(output_path, params, current_optimal, idx):
    """Save current state"""
    state_file = output_path / STATE_FILE
    state = {
        "params": params,
        "current_optimal": current_optimal,
        "idx": idx
    }
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def initialize_params(config, fresh_run):
    """Initialize parameters from config"""
    tunable = config["params"]["tunable"]
    tuned = config["params"]["tuned"]
    bounds = config["params"]["bounds"]
    other_params = config["params"]["other_params"]
    
    # Convert to internal format
    params = {}
    for key in tunable:
        params[key] = {
            "value": tunable[key],
            "tuned": 1 if tuned[key] else 0
        }
    
    # Get bounds
    param_bounds = {}
    for key in tunable:
        param_bounds[key] = bounds[key]
    
    return params, param_bounds, other_params

def generate_combinations(other_params):
    """Generate all combinations from other_params that have multiple values"""
    # Find parameters with multiple values
    list_params = {}
    single_params = {}
    
    for key, value in other_params.items():
        if isinstance(value, list):
            list_params[key] = value
        else:
            single_params[key] = value
    
    # Generate all combinations
    if list_params:
        keys = list(list_params.keys())
        values = [list_params[k] for k in keys]
        combinations = []
        for combo in product(*values):
            combo_dict = single_params.copy()
            for i, key in enumerate(keys):
                combo_dict[key] = combo[i]
            combinations.append(combo_dict)
        return combinations
    else:
        return [single_params]

def prepare_job_config(combo, params, param_name, param_val):
    """Prepare job configuration JSON - following test_communicate.py pattern"""
    # Start with combo (other_params with combinations)
    config = combo.copy()
    
    # Add tunable params (flatten like test_communicate.py)
    for key, value in params.items():
        if key == param_name:
            config[key] = param_val
        else:
            config[key] = value["value"]
    
    # Handle list values (shouldn't happen after generate_combinations, but just in case)
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 0:
            config[key] = value[0]
    
    return config

async def submit_all_jobs(exp_id, global_seed, param_name, itr_no, params, param_val, other_params):
    """Submit all job combinations to PSC"""
    combinations = generate_combinations(other_params)
    
    tasks = []
    job_ids = []
    job_counter = 0
    
    for combo in combinations:
        job_config = prepare_job_config(combo, params, param_name, param_val)
        # Mode is already in combo from other_params
        
        # Generate job ID: expId_global_seed_current_tunable_params_name_itr_no
        # Add counter to make unique for each combination
        job_id = f"{exp_id}_{global_seed}_{param_name}_{itr_no}_{job_counter}"
        job_ids.append(job_id)
        job_counter += 1
        
        # Submit job using psc.py - following test_communicate.py pattern
        # psc.slurm will parse --jobId= and call wrapper.py jobId=...
        command = f"sbatch psc.slurm {job_id}"
        task = submit_job(
            data_dict=job_config,
            command=command,
            jobname=f"{job_id}.json"
        )
        tasks.append((job_id, task))
        logger.info(f"Submitted job: {job_id}")
    
    # Wait for all submissions to complete
    for job_id, task in tasks:
        success, output = await task
        if success:
            logger.info(f"Job {job_id} submitted successfully: {output.strip()}")
        else:
            logger.error(f"Job {job_id} submission failed: {output}")
    
    return job_ids

async def wait_for_results(job_ids, check_interval=30):
    """Wait for all job results to arrive (waits indefinitely, checks every 30 seconds)"""
    results_dir = Path.home() / "data" / "raw" / "secondtuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Waiting for {len(job_ids)} jobs to complete...")
    
    while True:
        completed = []
        for job_id in job_ids:
            zip_file = results_dir / f"{job_id}.zip"
            
            # Check if zip file exists (result arrived)
            if zip_file.exists():
                completed.append(job_id)
        
        remaining = len(job_ids) - len(completed)
        logger.info(f"Completed: {len(completed)}/{len(job_ids)}, Remaining: {remaining}")
        
        if len(completed) == len(job_ids):
            logger.info("All jobs completed!")
            break
        
        await asyncio.sleep(check_interval)
    
    return completed

def unzip_and_extract_metrics(job_ids, objective):
    """Unzip results and extract metrics from summary.json"""
    results_dir = Path.home() / "data" / "raw" / "secondtuning"
    metrics = []
    
    for job_id in job_ids:
        zip_file = results_dir / f"{job_id}.zip"
        
        if not zip_file.exists():
            logger.warning(f"Zip file not found: {zip_file}")
            continue
        
        try:
            # Unzip - will automatically create job_id folder
            with zipfile.ZipFile(zip_file, 'r') as zipf:
                zipf.extractall(results_dir)
            
            # Read summary.json - zip contains job_id/filename structure
            summary_file = results_dir / job_id / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # Extract metric based on objective
                if objective == "hitrate":
                    metric = summary.get("temporal_hit_rate", summary.get("overall_hit_rate", 0))
                elif objective == "se":
                    metric = summary.get("speculation_efficiency", 0)
                else:
                    metric = summary.get("temporal_hit_rate", 0)
                
                metrics.append(metric)
                logger.info(f"Job {job_id}: {objective} = {metric}")
            else:
                logger.warning(f"summary.json not found in {results_dir / job_id}")
        except Exception as e:
            logger.error(f"Error processing {job_id}: {e}")
    
    return metrics

# Removed run_python_commands - now using async submission via psc.py

async def objective_function(exp_id, global_seed, params, param_name, param_val_left, param_val_mid, param_val_right, itr_no, other_params, objective):
    """Objective function for ternary search - submits jobs and calculates mean objective"""
    # Submit all combinations for three parameter values
    job_ids_left = await submit_all_jobs(exp_id, global_seed, param_name, f"{itr_no}_left", params, param_val_left, other_params)
    job_ids_mid = await submit_all_jobs(exp_id, global_seed, param_name, f"{itr_no}_mid", params, param_val_mid, other_params)
    job_ids_right = await submit_all_jobs(exp_id, global_seed, param_name, f"{itr_no}_right", params, param_val_right, other_params)
    
    # Wait for all results
    completed_left = await wait_for_results(job_ids_left)
    completed_mid = await wait_for_results(job_ids_mid)
    completed_right = await wait_for_results(job_ids_right)
    
    # Extract metrics and calculate mean
    metrics_left = unzip_and_extract_metrics(completed_left, objective)
    metrics_mid = unzip_and_extract_metrics(completed_mid, objective)
    metrics_right = unzip_and_extract_metrics(completed_right, objective)
    
    f_ml = sum(metrics_left) / len(metrics_left) if metrics_left else 0
    f_m = sum(metrics_mid) / len(metrics_mid) if metrics_mid else 0
    f_mr = sum(metrics_right) / len(metrics_right) if metrics_right else 0
    
    logger.info(f"Objective values - left: {f_ml:.4f}, mid: {f_m:.4f}, right: {f_mr:.4f}")
    
    return f_ml, f_m, f_mr

async def ternary_search(exp_id, global_seed, f, param_name, current_best, f_current_best, lower, upper, params, itr_base, other_params, objective):
    """Ternary search optimization"""
    history = []
    if current_best is not None and f_current_best is not None:
        history.append((current_best, f_current_best))
    
    left, right = lower, upper
    
    for i in range(MAX_ITR):
        start_time = time.time()
        
        if (right - left) < 0.001:  # Changed threshold for float values
            break
        
        mid = (left + right) / 2
        mid_left = (left + mid) / 2
        mid_right = (mid + right) / 2
        
        f_ml, f_m, f_mr = await f(exp_id, global_seed, params, param_name, mid_left, mid, mid_right, f"{itr_base}_{i}", other_params, objective)
        history.extend([(mid_left, f_ml), (mid, f_m), (mid_right, f_mr)])
        
        elapsed = time.time() - start_time
        logger.info(f"Ternary Search Iteration {i+1}: ml={mid_left:.4f}({f_ml:.4f}), m={mid:.4f}({f_m:.4f}), mr={mid_right:.4f}({f_mr:.4f}), time={elapsed:.2f}s")
        
        if f_ml > f_m:
            right = mid
        elif f_mr > f_m:
            left = mid
        else:
            left = mid_left
            right = mid_right
    
    best_x, best_val = max(history, key=lambda x: x[1])
    logger.info(f"All history for tuning {param_name}: {history}")
    logger.info(f"Best result for {param_name}: x={best_x:.4f}, val={best_val:.4f}")
    return best_x, best_val

async def run_experiment(config, output_path):
    """Main experiment loop"""
    global logger
    
    # Initialize
    fresh_run = config.get("fresh_run", True)
    global_seed = config.get("global_seed", 101)
    random.seed(global_seed)
    
    # Get objective from config
    exp_id = config.get("expId", "")
    objective = config.get("objective", "hitrate")
    
    # Initialize parameters from config
    params, param_bounds, other_params = initialize_params(config, fresh_run)
    
    # Load checkpoint or initialize
    checkpoint = None
    if not fresh_run:
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            # Restore state from checkpoint
            checkpoint_params = checkpoint.get("params", {})
            if checkpoint_params:
                # Update params with checkpoint values (preserving structure)
                for key in params:
                    if key in checkpoint_params:
                        params[key]["value"] = checkpoint_params[key].get("value", params[key]["value"])
                        params[key]["tuned"] = checkpoint_params[key].get("tuned", params[key]["tuned"])
            
            current_optimal = checkpoint.get("current_optimal", None)
            idx = checkpoint.get("idx", 0)
            logger.info(f"Resuming from checkpoint: idx={idx}, current_optimal={current_optimal}")
            logger.info(f"Restored parameters: {params}")
        else:
            logger.warning("fresh_run is False but no checkpoint found. Starting fresh.")
            current_optimal = None
            idx = 0
    else:
        # Fresh run - check if checkpoint exists and warn
        checkpoint = load_checkpoint(output_path)
        if checkpoint:
            logger.warning("fresh_run is True but checkpoint exists. Starting fresh (checkpoint will be overwritten).")
        current_optimal = None
        idx = 0
        logger.info("Starting fresh run")
    
    logger.info(f"Initial parameters: {params}")
    logger.info(f"Current optimal: {current_optimal}")
    
    # Main tuning loop
    try:
        while True:
            # Find untuned parameters
            untuned_param_list = [key for key in params if params[key]["tuned"] == 0]
            
            if len(untuned_param_list) == 0:
                logger.info("All parameters tuned. Experiment complete.")
                break
            
            # Pick random untuned parameter
            picked_param = random.choice(untuned_param_list)
            params[picked_param]["tuned"] = 1
            
            # Get bounds
            lower, upper = param_bounds[picked_param]
            
            logger.info(f"Tuning {picked_param}, bounds: ({lower}, {upper})")
            
            # Run ternary search
            best_x, best_val = await ternary_search(
                exp_id,
                global_seed,
                objective_function,
                picked_param,
                params[picked_param]["value"],
                current_optimal,
                lower,
                upper,
                params,
                f"{idx}_{picked_param}",
                other_params,
                objective
            )
            
            # Update state
            current_optimal = best_val
            params[picked_param]["value"] = best_x
            
            # Save checkpoint immediately after each parameter tuning
            checkpoint_state = {
                "params": {k: {"value": v["value"], "tuned": v["tuned"]} for k, v in params.items()},
                "current_optimal": current_optimal,
                "idx": idx + 1
            }
            save_checkpoint(output_path, checkpoint_state)
            save_state(output_path, params, current_optimal, idx + 1)
            
            logger.info(f"After Tuning {picked_param}: {best_x:.4f}, current_optimal: {current_optimal:.4f}")
            logger.info(f"Checkpoint saved. Can resume by setting fresh_run: false")
            idx += 1
    except (KeyboardInterrupt, Exception) as e:
        # Save checkpoint before exiting
        logger.error(f"Unexpected termination: {e}", exc_info=True)
        checkpoint_state = {
            "params": {k: {"value": v["value"], "tuned": v["tuned"]} for k, v in params.items()},
            "current_optimal": current_optimal,
            "idx": idx
        }
        save_checkpoint(output_path, checkpoint_state)
        save_state(output_path, params, current_optimal, idx)
        logger.info("Checkpoint saved before exit. Set fresh_run: false to resume.")
        raise

def main():
    """Main function"""
    global logger
    
    # Load config
    config = load_init_config()
    exp_id = config.get("expId", "default_exp")
    global_seed = config.get("global_seed", 101)
    
    # Setup paths
    output_path = setup_paths(exp_id, global_seed)
    logger = setup_logging(output_path)
    
    logger.info(f"Experiment started: {exp_id}, seed: {global_seed}")
    logger.info(f"Output path: {output_path}")
    
    # Run experiment
    try:
        asyncio.run(run_experiment(config, output_path))
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
