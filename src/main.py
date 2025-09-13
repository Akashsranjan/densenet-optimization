import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import time
import os
import csv
import psutil
import torch.nn.utils.prune as prune
from torch.cuda.amp import autocast
import copy
from typing import Dict, Any

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
BATCH_SIZES = [int(x) for x in os.environ.get("BATCH_SIZES", "1,4,8,16").split(',')]

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

def save_model_checkpoint(model: nn.Module, model_variant: str, optimization_technique: str, device: torch.device):
    """Saves the optimized model as a checkpoint."""
    models_dir = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    filename = f"{model_variant}_{optimization_technique}.pth"
    filepath = os.path.join(models_dir, filename)

    if isinstance(model, torch.jit.ScriptModule):
        model = model.cpu()
        torch.jit.save(model, filepath)
    else:
        torch.save(model.state_dict(), filepath)

    print(f"Model checkpoint saved to {filepath}")

def benchmark_model(model: nn.Module, batch_size: int, model_variant: str, device: torch.device, log_dir: str, optimization_technique: str, use_autocast: bool = False) -> Dict[str, Any]:
    """
    Benchmarks a given model for a specific batch size and logs results.
    """
    print(f"Benchmarking: {model_variant}, Batch Size: {batch_size}, Device: {device}, Optimization: {optimization_technique}")

    if use_autocast:
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    else:
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    profiler_dir = os.path.join(OUTPUT_DIR, "profiles", f"{model_variant}_{optimization_technique}")
    os.makedirs(profiler_dir, exist_ok=True)

    writer = SummaryWriter(os.path.join(log_dir, f"{model_variant}_{batch_size}_{optimization_technique}"))

    for _ in range(3):
        if use_autocast:
            with autocast():
                _ = model(dummy_input)
        else:
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
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        start_time = time.time()
        for step in range(5):
            with record_function(f"model_inference_batch_{step}"):
                if use_autocast:
                    with autocast():
                        _ = model(dummy_input)
                else:
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
        'optimization_technique': optimization_technique
    }

    return benchmark_data

def main():
    """Main function to run the entire benchmarking suite."""
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")

    optimizations = ["none", "fp16", "prune", "dynamic_quant", "script"]

    os.makedirs(os.path.join(LOG_DIR), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "profiles"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

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

    for opt in optimizations:
        try:
            print(f"\n--- Running {opt.upper()} Benchmark ---")
            
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            model.eval()

            # Set the device for this specific optimization
            current_device = DEVICE
            if opt == "dynamic_quant":
                current_device = torch.device("cpu")
            
            # Apply the specific optimization
            if opt == "none":
                model.to(current_device)
            elif opt == "fp16":
                model.to(DEVICE)
            elif opt == "prune":
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        prune.l1_unstructured(module, name="weight", amount=0.3)
                model.to(current_device)
            elif opt == "dynamic_quant":
                model = torch.quantization.quantize_dynamic(model.to(current_device), {nn.Linear}, dtype=torch.qint8)
            elif opt == "script":
                model = torch.jit.script(model)
                model.to(current_device)
            
            # Run the benchmark and append to the single CSV file
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for batch_size in BATCH_SIZES:
                    use_autocast = opt == "fp16"
                    results = benchmark_model(model, batch_size, f"densenet121_{opt}", current_device, LOG_DIR, opt, use_autocast)
                    writer.writerow(results)

            save_model_checkpoint(model, "densenet121", opt, current_device)

            print(f"Benchmarking complete for {opt}. Results written to {csv_path}")

        except Exception as e:
            print(f" Error during {opt} benchmark: {e}")
            print("Skipping this optimization...")

if __name__ == "__main__":
    main()