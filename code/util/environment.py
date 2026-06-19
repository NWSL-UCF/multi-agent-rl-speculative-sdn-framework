import importlib.metadata
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parents[2] / "requirements.txt"


def _parse_requirement_name(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return match.group(1) if match else None


def _lookup_installed_version(package_name):
    candidates = [
        package_name,
        package_name.replace("-", "_"),
        package_name.replace("_", "-"),
    ]
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def load_requirements(requirements_path=None):
    path = Path(requirements_path) if requirements_path else REQUIREMENTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"requirements.txt not found: {path}")

    packages = []
    with open(path) as f:
        for line in f:
            name = _parse_requirement_name(line)
            if name:
                packages.append(name)
    return packages


def _collect_compute_info(selected_device=None):
    info = {
        "cuda_available": False,
        "device": "cpu",
        "cuda_version": None,
        "gpu_name": None,
        "cudnn_version": None,
    }

    if selected_device:
        info["device"] = selected_device.split(":")[0] if selected_device.startswith("cuda") else selected_device

    try:
        import torch
    except ImportError:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_version"] = torch.version.cuda
    if info["cuda_available"]:
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        if selected_device is None:
            info["device"] = "cuda"
    elif selected_device is None:
        info["device"] = "cpu"

    return info


def collect_environment(requirements_path=None, selected_device=None):
    path = Path(requirements_path) if requirements_path else REQUIREMENTS_PATH
    packages = {}
    missing = []

    for name in load_requirements(path):
        version = _lookup_installed_version(name)
        if version is None:
            missing.append(name)
            packages[name] = None
        else:
            packages[name] = version

    return {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "requirements_file": str(path),
        "compute": _collect_compute_info(selected_device),
        "packages": packages,
        "missing_packages": missing,
    }


def save_environment_json(output_dir, requirements_path=None, selected_device=None, logger=None):
    environment = collect_environment(requirements_path, selected_device=selected_device)
    output_path = Path(output_dir) / "environments.json"
    with open(output_path, "w") as f:
        json.dump(environment, f, indent=2)
    if logger:
        logger.file_saved("Environment", 1)
    return output_path
