#!/usr/bin/env python3
"""Simple test script for communicate.py module"""

import json
import asyncio
from pathlib import Path
from psc import submit_job


def load_config_from_init():
    """Load and flatten config from init.json"""
    init_file = Path(__file__).parent / "init.json"
    
    with open(init_file, 'r') as f:
        data = json.load(f)
    
    # Extract tunable and other_params
    tunable = data.get("params", {}).get("tunable", {})
    other_params = data.get("params", {}).get("other_params", {})
    
    # Flatten into single dict
    config = {}
    config.update(tunable)
    config.update(other_params)
    
    # Handle list values (take first element)
    for key, value in config.items():
        if isinstance(value, list) and len(value) > 0:
            config[key] = value[0]
    
    return config


async def test_submit():
    # Load config from init.json
    config = load_config_from_init()
    print(f"Loaded config with {len(config)} parameters")
    
    # Submit job (non-blocking)
    task = submit_job(
        data_dict=config,
        command="sbatch psc.slurm test_config",
        jobname="test_config.json"
    )
    print("Submitted job (non-blocking)")
    
    # Await result
    success, output = await task
    print(f"Job - Success: {success}")
    print(f"Job - Output: {output}")


if __name__ == "__main__":
    asyncio.run(test_submit())

