import ctypes
import json
import platform
import re
import subprocess
from functools import wraps
from typing import Any, Dict, List
from warnings import warn

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 35

# Conversions from semantic version numbers
SEMVER_TO_CORES = {
    (1, 0): 8,  # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,  # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,  # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,  # Volta
    (7, 2): 64,
    (7, 5): 64,  # Turing
    (8, 0): 64,  # Ampere
    (8, 6): 64,
    (8, 7): 64,  # Ada Lovelace (RTX 40 series)
    (8, 9): 128,  # RTX 4060
}

SEMVER_TO_ARCH = {
    (1, 0): "tesla",
    (1, 1): "tesla",
    (1, 2): "tesla",
    (1, 3): "tesla",
    (2, 0): "fermi",
    (2, 1): "fermi",
    (3, 0): "kepler",
    (3, 2): "kepler",
    (3, 5): "kepler",
    (3, 7): "kepler",
    (5, 0): "maxwell",
    (5, 2): "maxwell",
    (5, 3): "maxwell",
    (6, 0): "pascal",
    (6, 1): "pascal",
    (6, 2): "pascal",
    (7, 0): "volta",
    (7, 2): "volta",
    (7, 5): "turing",
    (8, 0): "ampere",
    (8, 6): "ampere",
    (8, 7): "ada_lovelace",  # RTX 40 series
    (8, 9): "ampere",
}

MEMORY_BUS_WIDTH = {
    "tesla": 384,
    "fermi": 384,
    "kepler": 384,
    "maxwell": 384,
    "pascal": 384,
    "volta": 3072,
    "turing": 4096,
    "ampere": 6144,
    "ada_lovelace": 5120,  # RTX 40 series
}

DATA_RATE = {
    "tesla": 2,
    "fermi": 2,
    "kepler": 2,
    "maxwell": 2,
    "pascal": 2,
    "volta": 2,
    "turing": 2,
    "ampere": 2,
    "ada_lovelace": 2,  # RTX 40 series
}


# Decorator for CUDA API calls
def cuda_api_call(func):
    """
    Decorator to wrap CUDA API calls and check their results.
    Raises RuntimeError if the CUDA call does not return CUDA_SUCCESS.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError(
                f"{func.__name__} failed with error code {result}: {error_str.value.decode()}"
            )
        return result

    return wrapper


def cuda_api_call_warn(func):
    """
    Decorator to wrap CUDA API calls and check their results.
    Prints a warning message if the CUDA call does not return CUDA_SUCCESS.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            warn(
                f"Warning: {func.__name__} failed with error code {result}: {error_str.value.decode()}"
            )
        return result

    return wrapper


# Kiểm tra hệ điều hành hiện tại
def get_library_names():
    system = platform.system()
    if system == "Windows":
        return ["cuda.dll"]
    elif system == "Linux":
        return ["libcuda.so"]
    elif system == "Darwin":  # macOS
        return ["libcuda.dylib"]
    else:
        print("system", system)
        raise OSError("Unsupported operating system")


# Attempt to load the CUDA library
libnames = ("libcuda.so", "libcuda.dylib", "cuda.dll")
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    raise ImportError(f'Could not load any of: {", ".join(libnames)}')

# Thử tải thư viện CUDA phù hợp
# for libname in get_library_names():
#     try:
#         cuda = ctypes.CDLL(libname)
#     except OSError:
#         continue
#     else:
#         break
# else:
#     # Nếu không thể tải thư viện, báo lỗi
#     raise ImportError("Could not load CUDA library")


# CUDA API calls wrapped with the decorator
@cuda_api_call
def cuInit(flags):
    return cuda.cuInit(flags)


@cuda_api_call
def cuDeviceGetCount(count):
    return cuda.cuDeviceGetCount(count)


@cuda_api_call
def cuDeviceGet(device, ordinal):
    return cuda.cuDeviceGet(device, ordinal)


@cuda_api_call
def cuDeviceGetName(name, len, dev):
    return cuda.cuDeviceGetName(name, len, dev)


@cuda_api_call
def cuDeviceComputeCapability(major, minor, dev):
    return cuda.cuDeviceComputeCapability(major, minor, dev)


@cuda_api_call
def cuDeviceGetAttribute(pi, attrib, dev):
    return cuda.cuDeviceGetAttribute(pi, attrib, dev)


def get_bandwidth(memory_clock_rate, memory_bus_width, data_rate):
    # Convert memory clock rate from Hz to GHz for readability
    memory_clock_rate_ghz = memory_clock_rate / 1e6
    # Calculate memory bandwidth
    bandwidth = memory_clock_rate_ghz * memory_bus_width * data_rate / 8
    # Convert bandwidth from bytes per second to gigabytes per second
    bandwidth_gb_per_s = bandwidth
    return bandwidth_gb_per_s


def calculate_tflops(cuda_cores, gpu_clock_mhz):
    # Convert clock speed from MHz to GHz
    gpu_clock_ghz = gpu_clock_mhz / 1000.0
    # Calculate TFLOPS
    tflops = (cuda_cores * gpu_clock_ghz * 2) / 1000.0
    return tflops


@cuda_api_call_warn
def cuCtxCreate(pctx, flags, dev):
    try:
        result = cuda.cuCtxCreate_v2(pctx, flags, dev)
    except AttributeError:
        result = cuda.cuCtxCreate(pctx, flags, dev)
    return result


@cuda_api_call_warn
def cuMemGetInfo(free, total):
    try:
        result = cuda.cuMemGetInfo_v2(free, total)
    except AttributeError:
        result = cuda.cuMemGetInfo(free, total)
    return result


@cuda_api_call
def cuCtxDetach(ctx):
    return cuda.cuCtxDetach(ctx)


def get_gpu_info_from_nvidia_smi():
    output = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode("utf-8")
    gpu_info = []
    for line in output.split("\n"):
        match = re.match(r"GPU (\d+): (.*) \(UUID: (.*)\)", line)
        if match:
            gpu_id = int(match.group(1))
            gpu_name = match.group(2)
            gpu_uuid = match.group(3).replace("GPU-", "")
            gpu_info.append({"id": gpu_id, "name": gpu_name, "uuid": gpu_uuid})
    return gpu_info


def merge_gpu_info(specs, gpu_info):
    for spec in specs:
        for gpu in gpu_info:
            if spec.get("id") == gpu.get("id") or spec.get("uuid") == gpu.get("uuid"):
                spec.update(gpu)
    return specs


@cuda_api_call
def cuDriverGetVersion(version):
    return cuda.cuDriverGetVersion(version)


def get_pcie_info(i):
    try:
        # Get PCIe information using nvidia-smi
        nvidia_smi_output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if nvidia_smi_output.returncode == 0:
            pcie_info_lines = nvidia_smi_output.stdout.strip().split("\n")
            if i < len(pcie_info_lines):
                pcie_info = pcie_info_lines[i].split(", ")
                pcie_info_dict = {
                    "pcie_link_gen_max": int(pcie_info[0]),
                    "pcie_link_width_max": int(pcie_info[1]),
                }
                return pcie_info_dict
            else:
                return None
        else:
            return None
    except Exception as e:
        return None


def get_ubuntu_version() -> str:
    try:
        output = subprocess.check_output(["lsb_release", "-r"]).decode("utf-8")
        version = output.strip().split("\t")[-1]  # Lấy phiên bản từ output
        return version
    except subprocess.CalledProcessError:
        return 0


def get_cpu_cores() -> int:
    try:
        # Đọc thông tin về số lõi CPU từ /proc/cpuinfo
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
            cores = cpuinfo.count("processor")  # Đếm số lõi
        return cores
    except Exception as e:
        print(f"Error getting CPU cores: {e}")
        return 0


def get_total_cpu_cores() -> int:
    try:
        # Sử dụng lệnh lscpu để lấy tổng số lõi CPU
        lscpu_output = subprocess.check_output(["lscpu"]).decode("utf-8")
        for line in lscpu_output.split("\n"):
            if "CPU(s):" in line:
                return int(line.split()[1])
    except Exception as e:
        print(f"Error getting total CPU cores: {e}")
        return 0


def get_used_cpu_cores() -> int:
    try:
        # Sử dụng lệnh top để lấy thông tin về CPU
        top_output = subprocess.check_output(["top", "-bn1"]).decode("utf-8")

        # Tìm kiếm dòng chứa thông tin về CPU
        cpu_line = next(line for line in top_output.split("\n") if "Cpu(s)" in line)

        # Lấy phần trăm sử dụng CPU từ dòng CPU
        cpu_usage_match = re.search(r"(\d+\.\d+) id", cpu_line)
        if cpu_usage_match:
            cpu_idle = float(cpu_usage_match.group(1))
            cpu_usage_total = 100.0 - cpu_idle

            # Lấy số lõi CPU vật lý trên hệ thống
            num_physical_cores = get_total_cpu_cores()

            # Tính toán số lõi CPU đã sử dụng
            used_cpu_cores = round((cpu_usage_total / 100) * num_physical_cores)

            return used_cpu_cores
        else:
            print("Error parsing CPU usage")
            return 0
    except Exception as e:
        print(f"Error getting used CPU cores: {e}")
        return 0


def get_ram_info() -> dict:
    try:
        # Sử dụng lệnh free để lấy thông tin RAM
        free_output = subprocess.check_output(["free", "-m"]).decode("utf-8")
        lines = free_output.split("\n")
        mem_info = lines[1].split()

        total_ram = int(mem_info[1])
        used_ram = int(mem_info[2])
        free_ram = int(mem_info[3])

        return {
            "total_ram_mb": total_ram,
            "used_ram_mb": used_ram,
            "free_ram_mb": free_ram,
        }
    except Exception as e:
        print(f"Error getting RAM info: {e}")
        return {"total_ram_mb": 0, "used_ram_mb": 0, "free_ram_mb": 0}


def get_system_disk_info():
    try:
        # Use the df command to get disk information
        df_output = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True,
            text=True,
        )
        if df_output.returncode == 0:
            df_lines = df_output.stdout.strip().split("\n")
            if len(df_lines) > 1:
                headers = df_lines[0].split()
                values = df_lines[1].split()
                disk_info = dict(zip(headers, values))
                return disk_info
    except Exception as e:
        print(f"Error while getting system disk info: {e}")
    return {}


def get_all_disk_info():
    try:
        # Use the df command to get disk information for all disks
        df_output = subprocess.run(
            ["df", "-h"],
            capture_output=True,
            text=True,
        )
        if df_output.returncode == 0:
            df_lines = df_output.stdout.strip().split("\n")
            disk_info_list = []
            for line in df_lines[1:]:
                values = line.split()
                disk_info = {
                    "filesystem": values[0],
                    "size": values[1],
                    "used": values[2],
                    "avail": values[3],
                    "use_percentage": values[4],
                    "mounted_on": values[5],
                }
                disk_info_list.append(disk_info)
            return disk_info_list
    except Exception as e:
        print(f"Error while getting disk info: {e}")
    return []


# Main function
def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format
    {
        'id': int,
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float,
        'architecture': str,
        'cuda_cores': int,
        'memory_bus_width': int,
        'mem_bandwidth_gb_per_s': float,
        'uuid': str  # UUID of the GPU,
        'cuda_version': str,  # CUDA version
        'driver_version': str  # Driver version
    }
    """
    # Initialize CUDA
    cuInit(0)

    # Get CUDA version
    version = ctypes.c_int()
    cuDriverGetVersion(ctypes.byref(version))
    cuda_version = f"{version.value // 1000}.{(version.value % 1000) // 10}"
    driver_version_raw = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )
        .decode("utf-8")
        .strip()
    )
    driver_versions = driver_version_raw.split("\n")

    num_gpus = ctypes.c_int()
    cuDeviceGetCount(ctypes.byref(num_gpus))

    device_specs = []
    for i in range(num_gpus.value):
        pcie_info = get_pcie_info(i)
        driver_version = driver_versions[i].strip()

        spec = {
            "id": i,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
            "pcie_link_gen_max": pcie_info["pcie_link_gen_max"] if pcie_info else None,
            "pcie_link_width_max": (
                pcie_info["pcie_link_width_max"] if pcie_info else None
            ),
            "ubuntu_version": get_ubuntu_version(),
            "cpu_cores": get_cpu_cores(),
            "used_cores": get_used_cpu_cores(),
            "ram_info": get_ram_info(),
            "system_disk": get_system_disk_info(),
            "all_disk_info": get_all_disk_info(),
        }
        device = ctypes.c_int()
        cuDeviceGet(ctypes.byref(device), i)

        name = b" " * 100
        cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
        spec["name"] = name.split(b"\0", 1)[0].decode()

        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        cuDeviceComputeCapability(
            ctypes.byref(cc_major), ctypes.byref(cc_minor), device
        )
        compute_capability = (cc_major.value, cc_minor.value)
        spec["compute_capability"] = compute_capability

        cores = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
        )
        spec["cores"] = cores.value

        threads_per_core = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(threads_per_core),
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            device,
        )
        spec["concurrent_threads"] = cores.value * threads_per_core.value

        clockrate = ctypes.c_int()
        cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device
        )
        spec["gpu_clock_mhz"] = clockrate.value / 1000.0

        cuDeviceGetAttribute(
            ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device
        )
        spec["mem_clock_mhz"] = clockrate.value / 1000.0

        memory_bus_width = MEMORY_BUS_WIDTH.get(SEMVER_TO_ARCH[compute_capability], 0)
        spec["memory_bus_width"] = memory_bus_width

        memory_clockrate = ctypes.c_int()  # New: Memory Clock Rate
        cuDeviceGetAttribute(
            ctypes.byref(memory_clockrate),
            CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
            device,
        )

        spec["mem_bandwidth_gb_per_s"] = get_bandwidth(
            memory_clockrate.value,
            memory_bus_width,
            DATA_RATE[SEMVER_TO_ARCH[compute_capability]],
        )

        context = ctypes.c_void_p()
        if cuCtxCreate(ctypes.byref(context), 0, device) == CUDA_SUCCESS:
            free_mem = ctypes.c_size_t()
            total_mem = ctypes.c_size_t()

            cuMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))

            spec["total_mem_mb"] = total_mem.value / 1024**2
            spec["free_mem_mb"] = free_mem.value / 1024**2

            spec["architecture"] = SEMVER_TO_ARCH.get(compute_capability, 0)
            spec["cuda_cores"] = cores.value * SEMVER_TO_CORES.get(
                compute_capability, 0
            )

            spec["uuid"] = ""  # Placeholder for UUID to be merged later

            # Calculate TFLOPS
            spec["tflops"] = calculate_tflops(spec["cuda_cores"], spec["gpu_clock_mhz"])

            cuCtxDetach(context)

        device_specs.append(spec)
    return device_specs


if __name__ == "__main__":
    gpu_info = get_gpu_info_from_nvidia_smi()
    specs = get_cuda_device_specs()
    merged_specs = merge_gpu_info(specs, gpu_info)
    print(json.dumps(merged_specs, indent=2))

# python3.10 -m pip install nvidia-ml-py
# https://pypi.org/project/nvidia-ml-py/
# >>> from pynvml import *
# >>> nvmlInit()
# >>> print(f"Driver Version: {nvmlSystemGetDriverVersion()}")
# Driver Version: 11.515.48
# >>> deviceCount = nvmlDeviceGetCount()
# >>> for i in range(deviceCount):
# ...     handle = nvmlDeviceGetHandleByIndex(i)
# ...     print(f"Device {i} : {nvmlDeviceGetName(handle)}")
# ...
# Device 0 : Tesla K40c

# >>> nvmlShutdown()
