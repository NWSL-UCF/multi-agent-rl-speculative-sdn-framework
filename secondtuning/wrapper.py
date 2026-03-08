import sys
import json
import subprocess
import zipfile
import os
from pathlib import Path

SSH_KEY = "~/.ssh/home_desk"
SCP_HOST = "rouf@abdur-rouf.com:~/data/raw/secondtuning/"
SCP_PORT = "2222"


def parse_job_id():
    """Parse jobId from command line argument"""
    if len(sys.argv) < 2:
        print("Usage: python wrapper.py jobId=121321")
        sys.exit(1)
    
    arg = sys.argv[1]
    # Handle both --jobId= and jobId= formats
    if arg.startswith("--jobId="):
        job_id = arg.split("=", 1)[1]
    elif arg.startswith("jobId="):
        job_id = arg.split("=", 1)[1]
    else:
        print("Error: Argument must be in format 'jobId=121321' or '--jobId=121321'")
        sys.exit(1)
    
    return job_id


def load_json_config(job_id):
    """Load JSON configuration file"""
    jobs_dir = Path.home() / "data" / "raw" / "jobs"
    json_file = jobs_dir / f"{job_id}.json"
    
    if not json_file.exists():
        print(f"Error: Configuration file not found: {json_file}")
        sys.exit(1)
    
    with open(json_file, 'r') as f:
        return json.load(f)


def json_to_args(json_config):
    """Convert JSON config to command line arguments"""
    args = []
    for key, value in json_config.items():
        if value is not None:
            args.append(f"--{key}")
            args.append(str(value))
    return args


def run_main(base_path, json_config):
    """Run main.py with configuration from JSON"""
    # Convert JSON to command line arguments
    json_args = json_to_args(json_config)
    
    # Add base_path
    json_args.extend(["--base_path", base_path])
    
    # Run main.py
    cmd = [sys.executable, "sdn/main.py"] + json_args
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def create_zip(job_id, base_path):
    """Create zip file with required files including folder structure"""
    base_dir = Path(base_path)
    zip_path = base_dir.parent / f"{job_id}.zip"
    
    required_files = ["lti_metrics.csv", "info.log", "summary.json", "args.json"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in required_files:
            file_path = base_dir / filename
            if file_path.exists():
                # Include folder structure: job_id/filename
                arcname = f"{job_id}/{filename}"
                zipf.write(file_path, arcname)
                print(f"Added to zip: {arcname}")
            else:
                print(f"Warning: File not found: {file_path}")
    
    return zip_path


def scp_file(zip_path):
    """SCP zip file to remote server"""
    ssh_key_path = Path(SSH_KEY).expanduser()
    scp_cmd = [
        "scp",
        "-i", str(ssh_key_path),
        "-P", SCP_PORT,
        str(zip_path),
        SCP_HOST
    ]
    
    print(f"Transferring: {' '.join(scp_cmd)}")
    result = subprocess.run(scp_cmd)
    return result.returncode == 0


def main():
    """Main wrapper function"""
    # Parse job ID
    job_id = parse_job_id()
    print(f"Processing job ID: {job_id}")
    
    # Load JSON configuration
    json_config = load_json_config(job_id)
    print(f"Loaded configuration from: ~/data/raw/jobs/{job_id}.json")
    
    # Set base path
    base_path = str(Path.home() / "data" / "raw" / "secondtuning" / job_id)
    os.makedirs(base_path, exist_ok=True)
    print(f"Output directory: {base_path}")
    
    # Run main.py
    print("\n=== Running main.py ===")
    success = run_main(base_path, json_config)
    
    if not success:
        print("Error: main.py failed")
        sys.exit(1)
    
    # Create zip file
    print("\n=== Creating zip file ===")
    zip_path = create_zip(job_id, base_path)
    print(f"Created zip: {zip_path}")
    
    # Transfer zip file
    print("\n=== Transferring zip file ===")
    if scp_file(zip_path):
        print("Transfer successful!")
    else:
        print("Error: Transfer failed")
        sys.exit(1)
    
    print(f"\n=== Job {job_id} completed successfully ===")


if __name__ == "__main__":
    main()