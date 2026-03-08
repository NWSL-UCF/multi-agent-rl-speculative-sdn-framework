import os
import json
import signal
import logging

LOG_FILENAME = "__stop__.log"

def setup_logger():
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILENAME)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def stop_processes():
    # Load config.json to get expId
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if not os.path.exists(config_path):
        logging.error("config.json not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    exp_id = config.get("expId")
    if not exp_id:
        logging.error("expId not found in config.json.")
        return

    pid_file = os.path.join(os.path.dirname(__file__), exp_id, "pids.json")

    if not os.path.exists(pid_file):
        logging.warning(f"No PID file found at {pid_file}. Nothing to stop.")
        return

    with open(pid_file, "r") as f:
        pids = json.load(f)

    for name, pid in pids.items():
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info(f"Terminated {name} (PID: {pid})")
        except ProcessLookupError:
            logging.warning(f"{name} (PID: {pid}) was not running.")
        except Exception as e:
            logging.error(f"Failed to terminate {name} (PID: {pid}): {e}")

    os.remove(pid_file)
    logging.info("All tracked processes have been stopped and PID file removed.")

def main():
    setup_logger()
    stop_processes()

if __name__ == "__main__":
    main()
