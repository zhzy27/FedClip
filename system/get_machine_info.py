#!/usr/bin/env python3
"""
服务器环境信息检测脚本
用于获取类似论文中描述的硬件配置和软件环境
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

def run_command(cmd, shell=False):
    """执行命令并返回输出"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def get_gpu_info():
    """获取GPU信息"""
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    
    # 尝试使用nvidia-smi
    nvidia_smi = run_command("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", shell=True)
    if "Error" not in nvidia_smi and nvidia_smi:
        print("NVIDIA GPUs:")
        print(nvidia_smi)
    else:
        # 尝试其他方法
        lspci_gpu = run_command("lspci | grep -i vga", shell=True)
        if lspci_gpu:
            print("GPU devices:")
            print(lspci_gpu)
        else:
            print("No GPU information available or no NVIDIA GPU detected")

def get_cpu_info():
    """获取CPU信息"""
    print("\n" + "=" * 50)
    print("CPU Information")
    print("=" * 50)
    
    if platform.system() == "Linux":
        # Linux系统
        cpu_info = run_command("lscpu | grep -E '(Model name|Socket|Core|Thread|CPU\\(s\\))'", shell=True)
        if cpu_info:
            print(cpu_info)
        else:
            cpu_model = run_command("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True)
            if cpu_model:
                print(cpu_model)
    else:
        # 其他系统
        print(f"Architecture: {platform.machine()}")
        print(f"Processor: {platform.processor()}")

def get_memory_info():
    """获取内存信息"""
    print("\n" + "=" * 50)
    print("Memory Information")
    print("=" * 50)
    
    if platform.system() == "Linux":
        memory_info = run_command("free -h | grep -E '(total|Mem)'", shell=True)
        if memory_info:
            print(memory_info)
        
        # 尝试获取详细内存信息
        mem_details = run_command("dmidecode -t memory 2>/dev/null | grep -i size | grep -v 'No Module'", shell=True)
        if mem_details and "Error" not in mem_details:
            print("\nMemory modules:")
            print(mem_details)
    else:
        print("Memory information retrieval may be limited on this system")

def get_software_info():
    """获取软件环境信息"""
    print("\n" + "=" * 50)
    print("Software Environment")
    print("=" * 50)
    
    # Python信息
    print(f"Python version: {sys.version}")
    
    # 尝试导入PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch not installed")
    
    # 尝试获取系统CUDA版本
    cuda_version = run_command("nvcc --version 2>/dev/null | grep 'release'", shell=True)
    if cuda_version and "Error" not in cuda_version:
        print(f"System CUDA: {cuda_version}")
    
    # 操作系统信息
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")

def get_storage_info():
    """获取存储信息"""
    print("\n" + "=" * 50)
    print("Storage Information")
    print("=" * 50)
    
    if platform.system() == "Linux":
        storage_info = run_command("df -h / | tail -1", shell=True)
        if storage_info:
            print(f"Root filesystem: {storage_info}")
        
        # 检查是否有NVMe存储
        nvme_check = run_command("lsblk | grep -i nvme", shell=True)
        if nvme_check:
            print("NVMe drives detected")

def check_environment():
    """检查是否与论文环境匹配"""
    print("\n" + "=" * 50)
    print("Environment Check vs Paper Configuration")
    print("=" * 50)
    
    paper_spec = {
        "GPU": "NVIDIA GeForce RTX 4090 (24GB)",
        "CPU": "Intel Xeon Gold 5318Y",
        "RAM": "1.5TB",
        "PyTorch": "1.12.1",
        "CUDA": "11.6",
        "Python": "3.9"
    }
    
    print("Paper configuration:")
    for key, value in paper_spec.items():
        print(f"  {key}: {value}")
    
    print("\nYour system:")
    # 这里可以添加实际的比较逻辑
    print("  [请根据上方输出信息手动比较]")

def main():
    """主函数"""
    print("Server Environment Detection Script")
    print("This script will detect your server hardware and software configuration")
    print("Note: Some commands may require sudo privileges for detailed information\n")
    
    get_gpu_info()
    get_cpu_info()
    get_memory_info()
    get_storage_info()
    get_software_info()
    check_environment()
    
    print("\n" + "=" * 50)
    print("Detection Complete")
    print("=" * 50)
    print("\nNote: For complete memory module details, run with sudo privileges")
    print("Some information might require additional tools (dmidecode, nvidia-smi)")

if __name__ == "__main__":
    main()