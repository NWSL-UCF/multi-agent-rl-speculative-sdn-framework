import time
import threading
import fcntl
import logging
import os
from flask import Flask, jsonify, request
from datetime import datetime
import json
import pandas as pd
import argparse

app = Flask(__name__)

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CSV_FILE = ""
LOG_FILENAME = "server.log"

def createExpBaseDirectory(args):
    os.makedirs(os.path.join(BASE_DIR, args.expId), exist_ok=True)

def setup_log(args):
    LOG_FILE = os.path.join(BASE_DIR, args.expId, LOG_FILENAME)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )



LOCK = threading.Lock()

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_SERVED = "SERVED"
STATUS_DONE = "DONE"
STATUS_ABORTED = "ABORTED"


def getMessageAsJSONlist(messageStr):
    try:
        existing_messages = json.loads(messageStr)
        if not isinstance(existing_messages, list):
            existing_messages = []
    except json.JSONDecodeError:
        existing_messages = []
    return existing_messages

def format_timestamp(timestamp):
    if timestamp < 0:
        return "N/A"
    """Convert timestamp to human-readable format."""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else "N/A"

def load_jobs():
    try:
        with open(CSV_FILE, "r") as file:
            fcntl.flock(file, fcntl.LOCK_SH)  # Shared (read) lock
            df = pd.read_csv(
                file,
                dtype={
                    "id": int,
                    "request_timestamp": float,
                    "completion_timestamp": float,
                    "required_time": float,
                },
            )
            fcntl.flock(file, fcntl.LOCK_UN)  # Unlock
        return df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Error loading jobs: {e}")
        return []

def get_column_names():
    """Return a list of column names from the CSV file (with shared lock)."""
    try:
        with open(CSV_FILE, "r") as file:
            fcntl.flock(file, fcntl.LOCK_SH)
            df = pd.read_csv(file, nrows=0)
            fcntl.flock(file, fcntl.LOCK_UN)
        return df.columns.tolist()
    except Exception as e:
        logging.error(f"Error reading CSV headers: {e}")
        return []

def save_jobs(jobs):
    """Save jobs back to the CSV file, using the CSV’s own header order."""
    try:
        columns = get_column_names()
        if not columns:
            logging.error("No columns found in CSV — aborting save.")
            return

        df = pd.DataFrame(jobs).reindex(columns=columns).fillna("")

        with open(CSV_FILE, "w", newline="") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            df.to_csv(file, index=False)
            fcntl.flock(file, fcntl.LOCK_UN)

        logging.info("Jobs successfully saved to CSV.")
    except Exception as e:
        logging.error(f"Error saving jobs: {e}")

@app.route("/request_job", methods=["POST"])
def request_job():
    """Assign a NOT_STARTED job to a requester and mark it as SERVED (using pandas)."""
    data = request.json or {}
    requested_by = data.get("requested_by")

    if not requested_by:
        logging.warning("Job request failed: No requester identification provided.")
        return jsonify({"error": "Requester identification is required"}), 400

    with LOCK:
        # Load into a DataFrame
        df = pd.DataFrame(load_jobs())

        # Find the first NOT_STARTED job
        mask = df["status"] == STATUS_NOT_STARTED
        if not mask.any():
            logging.info("No NOT_STARTED jobs available.")
            return jsonify({"error": "No available jobs"}), 404

        idx = mask.idxmax()
        timestamp = time.time()

        # Update fields
        df.at[idx, "requested_by"] = requested_by
        df.at[idx, "status"] = STATUS_SERVED
        df.at[idx, "request_timestamp"] = timestamp

        messages = getMessageAsJSONlist(df.at[idx, "message"])
        messages.append({
            "reason": f"{requested_by} requests this job for execution",
            "timestamp": timestamp
        })
        df.at[idx, "message"] = json.dumps(messages)

        # Persist changes
        save_jobs(df.to_dict(orient="records"))

        job_id = int(df.at[idx, "id"])
        job_parameters = json.loads(df.at[idx, "parameters"])

    logging.info(f"Job {job_id} assigned to {requested_by} and marked as SERVED.")
    return jsonify({"job_id": job_id, "paramters": job_parameters, "status": STATUS_SERVED}), 200


@app.route("/update_job_status", methods=["POST"])
def update_job_status():
    """Update job status as DONE or ABORTED (using pandas)."""
    data = request.json or {}
    job_id = data.get("job_id")
    status = data.get("status")
    message = data.get("message", "")

    if not isinstance(job_id, int) or status not in [STATUS_DONE, STATUS_ABORTED]:
        logging.warning(f"Invalid job status update request: job_id={job_id}, status={status}")
        return jsonify({"error": "Invalid job_id or status"}), 400

    with LOCK:
        df = pd.DataFrame(load_jobs())

        mask = (df["id"] == job_id) & (df["status"] == STATUS_SERVED)
        if not mask.any():
            return jsonify({"error": "Job not found or not in SERVED status"}), 404

        idx = mask.idxmax()
        now = time.time()

        df.at[idx, "status"] = status
        df.at[idx, "completion_timestamp"] = now
        df.at[idx, "required_time"] = now - df.at[idx, "request_timestamp"]

        msgs = getMessageAsJSONlist(df.at[idx, "message"])
        msgs.append({
            "reason": message if message else "No reason provided",
            "timestamp": now
        })
        df.at[idx, "message"] = json.dumps(msgs)

        save_jobs(df.to_dict(orient="records"))

    if status == STATUS_DONE:
        logging.info(f"Job {job_id} marked as DONE. Required time: {df.at[idx, 'required_time']} seconds.")
    else:
        logging.info(f"Job {job_id} ABORTED. Reason: {message or 'No reason provided'}.")

    return jsonify({"message": f"Job {job_id} updated to {status}", "job_id": job_id}), 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Flask server")
    parser.add_argument("--host", default="0.0.0.0", help="IP address to bind to")
    parser.add_argument("--jobDB", default="jobs.csv", help="CSV file (<filename>.csv) placed in the same directory as server.py")
    parser.add_argument("--port", type=int, default=5000, help="Port number to listen on")
    parser.add_argument("--expId", type=str, default="sim1", help="Give an unique name")
    args = parser.parse_args()
    createExpBaseDirectory(args)
    setup_log(args)
    logging.info(f"Starting Flask server on {args.host}:{args.port}...")
    CSV_FILE = os.path.join(BASE_DIR, args.expId, args.jobDB)
    app.run(host=args.host, port=args.port)  


# python server.py --expId=sim1 --jobDB=jobs.csv --host=0.0.0.0 --timeoutLimit=5050