import pandas as pd
import time
import fcntl  # For file locking on Linux/macOS
import os
import logging
import json
from math import floor
import argparse


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_FILE = ""
LOG_FILENAME = "job_cleaner.log"

def createExpBaseDirectory(args):
    os.makedirs(os.path.join(BASE_DIR, args.expId), exist_ok=True)

def setup_log(args):
    LOG_FILE = os.path.join(BASE_DIR, args.expId, LOG_FILENAME)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Job statuses
STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_SERVED = "SERVED"
STATUS_DONE = "DONE"
STATUS_ABORTED = "ABORTED"

# Timeout for SERVED jobs (90 seconds for testing, can be increased)
TIMEOUT_LIMIT = 24 * 60 * 60  # seconds

# How often the cleanup should run (e.g., every 30 minutes)
CLEANUP_INTERVAL = 30 * 60  # seconds

def get_message_as_json_list(message_str):
    """Convert a message string to a JSON list."""
    try:
        existing_messages = json.loads(message_str)
        if not isinstance(existing_messages, list):
            existing_messages = []
    except json.JSONDecodeError:
        existing_messages = []
    return existing_messages

def load_jobs():
    """Load jobs from the CSV file using pandas with file locking."""
    try:
        with open(CSV_FILE, "r") as file:
            fcntl.flock(file, fcntl.LOCK_SH)  # Shared lock (read mode)
            jobs_df = pd.read_csv(file)
            fcntl.flock(file, fcntl.LOCK_UN)  # Unlock the file

        logging.info("Successfully loaded jobs from CSV.")
        return jobs_df
    except Exception as e:
        logging.error(f"Error loading jobs: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame on error

def save_jobs(jobs_df):
    """Save jobs back to the CSV file using pandas with file locking."""
    try:
        with open(CSV_FILE, "w") as file:
            fcntl.flock(file, fcntl.LOCK_EX)  # Exclusive lock (write mode)
            jobs_df.to_csv(file, index=False)
            fcntl.flock(file, fcntl.LOCK_UN)  # Unlock the file

        logging.info("Successfully saved jobs to CSV.")
    except Exception as e:
        logging.error(f"Error saving jobs: {str(e)}")

def cleanup_jobs():
    """Reset ABORTED jobs and handle SERVED jobs that exceeded the time limit."""
    jobs_df = load_jobs()
    if jobs_df.empty:
        logging.info("No jobs to clean up.")
        return

    current_time = time.time()
    jobs_updated = False

    # Reset ABORTED jobs to NOT_STARTED
    aborted_mask = jobs_df["status"] == STATUS_ABORTED
    if aborted_mask.any():
        logging.info(f"Resetting {aborted_mask.sum()} ABORTED jobs to NOT_STARTED.")
        jobs_df.loc[aborted_mask, ["status", "requested_by", "request_timestamp", "completion_timestamp", "required_time"]] = [
            STATUS_NOT_STARTED, "", 0, 0, 0
        ]
        
        # Update messages
        for index in jobs_df[aborted_mask].index:
            prev_requester = jobs_df.at[index, "requested_by"]
            existing_messages = get_message_as_json_list(jobs_df.at[index, "message"])
            new_message = {
                "reason": f"Job Cleaner updated the job status to NOT STARTED due to previous failure at {prev_requester}!",
                "timestamp": current_time
            }
            existing_messages.append(new_message)
            jobs_df.at[index, "message"] = json.dumps(existing_messages)

        jobs_updated = True

    # Reset SERVED jobs older than TIMEOUT_LIMIT
    served_mask = (jobs_df["status"] == STATUS_SERVED) & (current_time - jobs_df["request_timestamp"] > TIMEOUT_LIMIT)
    if served_mask.any():
        logging.info(f"Resetting {served_mask.sum()} SERVED jobs due to timeout.")
        hours = floor(TIMEOUT_LIMIT / 3600)
        minutes = round((TIMEOUT_LIMIT % 3600) / 60)

        for index in jobs_df[served_mask].index:
            prev_requester = jobs_df.at[index, "requested_by"]
            existing_messages = get_message_as_json_list(jobs_df.at[index, "message"])
            new_message = {
                "reason": f"Job Cleaner updated the job status to NOT STARTED because the server isn't hearing back from {prev_requester} for {hours}:{minutes} hours!",
                "timestamp": current_time
            }
            existing_messages.append(new_message)
            jobs_df.at[index, "message"] = json.dumps(existing_messages)

        jobs_df.loc[served_mask, ["status", "requested_by", "request_timestamp", "completion_timestamp", "required_time"]] = [
            STATUS_NOT_STARTED, "", 0, 0, 0
        ]

        jobs_updated = True

    if jobs_updated:
        save_jobs(jobs_df)
        logging.info("Job cleanup completed successfully.")
    else:
        logging.info("No jobs required cleanup.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start job_cleaner background service")
    parser.add_argument("--jobDB", default="jobs.csv", help="CSV file (<filename>.csv) placed in the same directory as server.py")
    parser.add_argument("--expId", type=str, default="sim1", help="Give an unique name of your experiment")
    parser.add_argument("--cleanupInterval", type=int, default=1800, help="How often should aborted jobs be reset? (in seconds)")
    parser.add_argument("--timeoutLimit", type=int, default=86400, help="How often should served jobs be reset? (in seconds)")
    args = parser.parse_args()
    createExpBaseDirectory(args)
    setup_log(args)
    CSV_FILE = os.path.join(BASE_DIR, args.expId, args.jobDB)
    CLEANUP_INTERVAL = args.cleanupInterval
    TIMEOUT_LIMIT = args.timeoutLimit
    while True:
        cleanup_jobs()
        logging.info(f"Waiting {CLEANUP_INTERVAL} seconds before next cleanup cycle...")
        time.sleep(CLEANUP_INTERVAL)  # Wait before running cleanup again


# python job_cleaner.py --expId=sim1 --jobDB=jobs.csv --cleanupInterval=1800 --timeoutLimit=86400