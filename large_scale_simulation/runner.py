import requests
import os
import pandas as pd
import subprocess
import logging
import argparse

def setup_logger(instance, expId):
    log_path = os.path.expanduser(f"~/data/raw/{expId}/runner/{instance}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = "http://abdur-rouf.com:5000"
DONE = "DONE"
ABORTED = "ABORTED"

def update_job_status(job_id, status, message=""):
    url = f"{BASE_URL}/update_job_status"
    payload = {
        "job_id": job_id,
        "status": status,
        "message": message
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def request_job(requested_by):
    url = f"{BASE_URL}/request_job"
    payload = {"requested_by": requested_by}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    print(response)
    return response.json()

def getAllCombinations():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    csv_file = os.path.join(script_dir, "cmd.csv")
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def main(instance, expId):
    """Main function that executes jobs in a loop for a given instance and experiment ID."""
    # instance =  "hello"
    # expId = "world"
    # Set up logging for this instance and experiment ID
    setup_logger(instance, expId)
    logging.info("Starting job execution loop...")

    # Load the CSV file containing job commands
    commands = getAllCombinations()
    if commands is None:
        logging.error("Failed to load job commands from CSV. Exiting.")
        return

    # Request a job from the server
    jobinfo = request_job(instance)

    # Continue executing jobs as long as new ones are available
    while "job_id" in jobinfo:
        job_id = jobinfo["job_id"]

        # Fetch the corresponding command from the CSV using job_id
        try:
            python_command = commands.iloc[job_id]["command"]
        except IndexError:
            logging.error(f"Invalid job_id {job_id}. Skipping job.")
            jobinfo = request_job(instance)
            continue

        # Append instance information to the command
        python_command += f" --ins={instance}"

        try:
            logging.info(f"Executing command: {python_command}")

            # Run the command and wait for it to complete
            result = subprocess.run(
                python_command, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )

            logging.info(f"Job {job_id} completed successfully:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            # Update job status in case of failure
            error_message = f"Execution failed at {instance}: {e.stderr}"
            update_job_status(job_id, ABORTED, error_message)
            logging.error(f"Job {job_id} aborted due to error: {error_message}")

        # Request the next job
        jobinfo = request_job(instance)

    logging.info("No remaining jobs. Exiting job execution loop.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True, help="Instance name")
    parser.add_argument("--expId", required=True, help="Simulation ID (sim0, sim1, ...)")
    args = parser.parse_args()
    main(args.instance, args.expId)
