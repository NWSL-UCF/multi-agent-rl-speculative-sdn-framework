import argparse
import logging
import os
import random
import time
from datetime import datetime
import pandas as pd

# Setup arguments parser
parser = argparse.ArgumentParser(description="Simple test simulation")
parser.add_argument('--total_duration', type=int, default=10, help='Total duration for the simulation loop (seconds)')
parser.add_argument('--sleep_time', type=int, default=1, help='Sleep time between iterations (seconds)')
parser.add_argument('--processId', type=str, default='simulation_log', help='Process ID or name for logfile')
args = parser.parse_args()

# Setup directories
os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configure logging
log_file = os.path.join('logs', f'{args.processId}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# List of 10 sample strings
sample_texts = [
    "Hello", "Goodbye", "Welcome", "Thanks", "Yes",
    "No", "Maybe", "Confirmed", "Canceled", "Retry"
]

results = []
idx = 0
start_time = time.time()

logging.info("Simulation started")

while (time.time() - start_time) < args.total_duration:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selected_text = random.choice(sample_texts)

    result = {
        "idx": idx,
        "current_time": current_time,
        "selected_text": selected_text
    }

    results.append(result)

    logging.info(f"Iteration {idx}: {result}")

    idx += 1
    time.sleep(args.sleep_time)

logging.info("Simulation finished")

# Write results to CSV using pandas
csv_file = os.path.join('results', f'results_{args.processId}.csv')
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

logging.info(f"Results saved to {csv_file}")
