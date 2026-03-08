import argparse
import subprocess
import time

def main(start_idx, end_idx, instance, expId):
    processes = []
    for idx in range(start_idx, end_idx + 1):
        cmd = ["python", "runner.py", f"--expId={expId}", f"--instance={instance}_{idx}"]
        print(f"Starting process: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append(p)

    # Optionally wait for all to finish
    for p in processes:
        p.wait()
    print("All processes completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True, help="Start index")
    parser.add_argument("--end_idx", type=int, required=True, help="End index")
    parser.add_argument("--instance", type=str, required=True, help="Give a name of your machine")
    parser.add_argument("--expId", type=str, required=True, help="Give an Id of this experiment")
    args = parser.parse_args()
    main(args.start_idx, args.end_idx, args.instance, args.expId)