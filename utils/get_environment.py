import platform
import subprocess
import re
import torch
import sys

try:
    import psutil
except ImportError:
    print("psutil is not installed. Please install it using `pip install psutil`")
    sys.exit(1)


def get_cpu_info():
    cpu_info = {
        "model_name": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_cores": psutil.cpu_count(logical=True),
        "max_frequency": psutil.cpu_freq().max
    }
    return cpu_info


def get_memory_info():
    virtual_memory = psutil.virtual_memory()
    memory_info = {
        "total_memory": virtual_memory.total // (1024 ** 3)  # Convert bytes to GB
    }
    return memory_info


def get_gpu_info():
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                                           shell=True).decode("utf-8").strip()
        gpu_list = []
        for line in gpu_info.split("\n"):
            name, memory = line.split(", ")
            gpu_list.append({
                "name": name,
                "total_memory": int(memory) // 1024  # Convert MB to GB
            })
        return gpu_list
    except Exception as e:
        return []


def get_os_info():
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version()
    }
    return os_info


def get_cuda_version():
    try:
        cuda_version = subprocess.check_output("nvcc --version", shell=True).decode("utf-8")
        version = re.search(r"release (\d+\.\d+)", cuda_version).group(1)
        driver_version = \
        subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True).decode(
            "utf-8").strip().split('\n')[0]
        return version, driver_version
    except Exception as e:
        return None, None


def get_pytorch_version():
    return torch.__version__


def generate_description(cpu_info, memory_info, gpu_info, os_info, cuda_version, pytorch_version):
    gpu_count = len(gpu_info)
    if gpu_count > 0:
        gpu_details = ', '.join([f"{gpu['name']} with {gpu['total_memory']}GB VRAM" for gpu in gpu_info])
    else:
        gpu_details = "No NVIDIA GPUs"

    if cuda_version[0] is not None:
        cuda_version_str = f"CUDA Version {cuda_version[0]} (Driver {cuda_version[1]})"
    else:
        cuda_version_str = "No CUDA support"

    description = (
        f"All of our experiments, including the original trainings and evaluations of our models, have been run on a server with "
        f"{cpu_info['total_cores']} CPUs {cpu_info['model_name']} at {cpu_info['max_frequency']:.2f}GHz, "
        f"{memory_info['total_memory']}GB RAM and {gpu_count} NVIDIA {gpu_details}. "
        f"The operating system is {os_info['system']} {os_info['release']}, with {cuda_version_str} "
        f"and PyTorch {pytorch_version}."
    )
    return description


if __name__ == "__main__":
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)

    cpu_info = get_cpu_info()
    memory_info = get_memory_info()
    gpu_info = get_gpu_info()
    os_info = get_os_info()
    cuda_version = get_cuda_version()
    pytorch_version = get_pytorch_version()

    description = generate_description(cpu_info, memory_info, gpu_info, os_info, cuda_version, pytorch_version)
    print(description)
