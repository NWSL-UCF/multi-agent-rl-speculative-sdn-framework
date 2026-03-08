import multiprocessing
import itertools
import subprocess

def run_simulation(total_time, sleep_time, idx):
    process_id = f"sim_{idx}"
    command = [
        "python", "sim.py",
        "--total_duration", str(total_time),
        "--sleep_time", str(sleep_time),
        "--processId", process_id
    ]
    subprocess.run(command)

if __name__ == "__main__":
    parameters = {
        "total_time": [45, 60, 75],
        "sleep_time": [3, 5]
    }
    
    param_combinations = list(itertools.product(parameters["total_time"], parameters["sleep_time"]))
    
    processes = []
    for idx, (total_time, sleep_time) in enumerate(param_combinations):
        p = multiprocessing.Process(target=run_simulation, args=(total_time, sleep_time, idx))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    print("All simulations completed.")
