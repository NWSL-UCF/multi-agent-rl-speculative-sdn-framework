import pandas as pd
import argparse
import os
import logging
import json
from itertools import product

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_FILE = ""
LOG_FILENAME = "create_job_db.log"

def createExpBaseDirectory(args):
    os.makedirs(os.path.join(BASE_DIR, args.expId), exist_ok=True)

def setup_log(args):
    LOG_FILE = os.path.join(BASE_DIR, args.expId, LOG_FILENAME)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    

def generate_csv(filename, parameters_dict):
    parameters_keys = list(parameters_dict.keys())
    parameters_values = list(parameters_dict.values())
    parameters_list = [dict(zip(parameters_keys, combination)) for combination in product(*parameters_values)]
    total_jobs = len(parameters_list)
    
    df = pd.DataFrame({
        "id": range(total_jobs),
        "requested_by": ["" for _ in range(total_jobs)],
        "request_timestamp": [0 for _ in range(total_jobs)],
        "completion_timestamp": [0 for _ in range(total_jobs)],
        "required_time": [0 for _ in range(total_jobs)],
        "status": ["NOT_STARTED" for _ in range(total_jobs)],
        "message": ["[]" for _ in range(total_jobs)],
        "parameters": parameters_list
    })
    df.to_csv(filename, index=False)
    logging.info(f"CSV file '{filename}' generated with {total_jobs} jobs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job CSV file")
    parser.add_argument("--jobDB", default="jobs.csv", help="CSV file (<filename>.csv) placed in the same directory as server.py")
    parser.add_argument("--expId", type=str, default="sim1", help="Give a unique name")
    parser.add_argument('--parameters', type=str, required=True)
    args = parser.parse_args()

    createExpBaseDirectory(args)
    setup_log(args)

    parameters_dict = json.loads(args.parameters)
    CSV_FILE = os.path.join(BASE_DIR, args.expId, args.jobDB)
    generate_csv(CSV_FILE, parameters_dict)

    logging.info("Job database setup complete.")
    

# python create_job_db.py --total_jobs=100 --expId=sim1 --jobDB=jobs.csv