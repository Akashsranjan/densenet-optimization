import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import time
import os
import csv
import psutil
from typing import List, Dict, Any


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,4,8,16,32").split(',')]


def get_device() -> torch.device:
    """Detect and return the appropriate device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def get_system_metrics() -> Dict[str, float]:
    """Collects system-level memory and CPU metrics."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'ram_usage_mb': mem_info.rss / (1024 ** 2),
        'cpu_utilization_pct': psutil.cpu_percent(interval=1)
    }

def get_gpu_metrics() -> Dict[str, float]:
    """Collects GPU-specific memory and utilization metrics."""
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            vram_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            pynvml.nvmlShutdown()

            return {
                'vram_usage_mb': vram_used / (1024 ** 2),
                'vram_peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'gpu_utilization_pct': gpu_util
            }
        except Exception as e:
            # Fallback in case pynvml is not available or an error occurs
            print(f"Warning: Could not get full GPU metrics with pynvml. Reason: {e}")
            return {
                'vram_usage_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'vram_peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'gpu_utilization_pct': 0
            }
    return {
        'vram_usage_mb': 0,
        'vram_peak_mb': 0,
        'gpu_utilization_pct': 0
    }

def get_model_size_mb(model: nn.Module) -> float:
    """Calculates the size of the model in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024**2)


# Benchmarking Core Function
def benchmark_model(model: nn.Module, batch_size: int, model_variant: str, device: torch.device, output_dir: str, log_dir: str) -> Dict[str, Any]:
    """
    Benchmarks a given model for a specific batch size and logs results.
    """
    print(f"Benchmarking: {model_variant}, Batch Size: {batch_size}, Device: {device}")

    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    writer = SummaryWriter(os.path.join(log_dir, f"{model_variant}_{batch_size}"))

    for _ in range(3):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if device.type == 'cuda' else ProfilerActivity.CPU
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir, f"{model_variant}_{batch_size}")),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        start_time = time.time()
        for step in range(5):
            with record_function(f"model_inference_batch_{step}"):
                _ = model(dummy_input)
            prof.step()
        end_time = time.time()

    avg_latency_ms = (end_time - start_time) * 1000 / 5
    throughput_samples_sec = (batch_size * 5) / (end_time - start_time)

    system_metrics = get_system_metrics()
    gpu_metrics = get_gpu_metrics()

    writer.add_scalar("Latency/ms", avg_latency_ms)
    writer.add_scalar("Throughput/samples_per_sec", throughput_samples_sec)
    writer.add_scalar("Memory/VRAM_peak_mb", gpu_metrics['vram_peak_mb'])
    writer.close()

    accuracy_top1 = "N/A"
    accuracy_top5 = "N/A"

    benchmark_data = {
        'model_variant': model_variant,
        'batch_size': batch_size,
        'device': str(device),
        'ram_usage_mb': system_metrics['ram_usage_mb'],
        'vram_usage_mb': gpu_metrics['vram_usage_mb'],
        'cpu_utilization_pct': system_metrics['cpu_utilization_pct'],
        'gpu_utilization_pct': gpu_metrics['gpu_utilization_pct'],
        'latency_ms': avg_latency_ms,
        'throughput_samples_sec': throughput_samples_sec,
        'accuracy_top1': accuracy_top1,
        'accuracy_top5': accuracy_top5,
        'model_size_mb': get_model_size_mb(model),
        'optimization_technique': 'None'
    }

    return benchmark_data


def main():
    """Main function to run the entire benchmarking suite."""
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    print("Loading baseline DenseNet-121 model...")
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(DEVICE)

    csv_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    fieldnames = [
        'model_variant', 'batch_size', 'device', 'ram_usage_mb', 'vram_usage_mb',
        'cpu_utilization_pct', 'gpu_utilization_pct', 'latency_ms',
        'throughput_samples_sec', 'accuracy_top1', 'accuracy_top5',
        'model_size_mb', 'optimization_technique'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for batch_size in BATCH_SIZES:
        results = benchmark_model(model, batch_size, "densenet121_baseline", DEVICE, OUTPUT_DIR, os.path.join(LOG_DIR, "tensorboard"))
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results)

    print("\nBaseline benchmarking complete. Results saved to benchmark_results.csv and TensorBoard logs.")

if __name__ == "__main__":
    main()