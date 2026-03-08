import json
import tempfile
import asyncio
from pathlib import Path

REMOTE_HOST = "psc"
SSH_KEY = "~/.ssh/psc"
REMOTE_DIR = "/jet/home/arouf/data/raw/jobs"

async def transfer_file(file_path, remote_host=REMOTE_HOST, remote_dir=REMOTE_DIR, ssh_key=SSH_KEY):
    if not Path(file_path).exists():
        return False
    try:
        process = await asyncio.create_subprocess_exec(
            "rsync", "-avzP", "-e", f"ssh -i {ssh_key}",
            str(file_path), f"{remote_host}:{remote_dir}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        return await process.wait() == 0
    except Exception:
        return False

async def submit_command(command, remote_host=REMOTE_HOST, ssh_key=SSH_KEY):
    try:
        process = await asyncio.create_subprocess_exec(
            "ssh", "-i", ssh_key, remote_host, command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        stdout = stdout.decode('utf-8') if stdout else ""
        stderr = stderr.decode('utf-8') if stderr else ""
        return (True, stdout) if process.returncode == 0 else (False, stderr)
    except Exception as e:
        return False, str(e)


async def submit_dict_and_command(data_dict, command, remote_host=REMOTE_HOST, remote_dir=REMOTE_DIR, jobname="temp_config.json", ssh_key=SSH_KEY):
    temp_file = Path(tempfile.gettempdir()) / jobname
    try:
        with open(temp_file, 'w') as f:
            json.dump(data_dict, f, indent=2)
        if not await transfer_file(str(temp_file), remote_host, remote_dir, ssh_key):
            return False, "File transfer failed"
        return await submit_command(command, remote_host, ssh_key)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def submit_job(data_dict, command, remote_host=REMOTE_HOST, remote_dir=REMOTE_DIR, jobname="temp_config.json", ssh_key=SSH_KEY):
    return asyncio.create_task(submit_dict_and_command(data_dict, command, remote_host, remote_dir, jobname, ssh_key))