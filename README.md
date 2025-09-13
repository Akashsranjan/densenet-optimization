Based on your provided requirements, here is a complete and structured documentation for your assignment's `README.md` file.

### `README.md`

# Deep Learning Model Optimization Benchmark

## 1\. Project Overview

This project provides a comprehensive benchmarking suite to evaluate the performance of a pre-trained `DenseNet-121` deep learning model under various optimization techniques. The primary objective is to measure and compare key metrics—such as latency, throughput, and memory consumption—to determine the most effective optimization approach for this model on a GPU-enabled environment. The project is designed to be fully reproducible using Docker and `docker-compose`.

## 2\. Setup Instructions

To get the project up and running, ensure you have the following software installed:

  - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support)

<!-- end list -->

1.  **Clone the repository:**

    ```bash
    git clone [your_repo_url]
    cd densenet-optimization
    ```

2.  **Build and run the benchmark:** The included script automates the Docker image build process and starts the benchmark container.

## 3\. Usage Guide

To run the full benchmark suite and generate the results, simply execute the `build_and_run.sh` script from the project's root directory:

```bash
./build_and_run.sh
```

This script will:

1.  Build the Docker image with the latest code.
2.  Run the `benchmark-app` container, which executes the `main.py` script.
3.  The script will perform benchmarks for all optimizations and all batch sizes.
4.  All results will be saved to a single CSV file at `results/benchmark_results.csv`.
5.  Performance traces will be saved for analysis in TensorBoard logs within the `logs/` directory.

## 4\. Optimization Approaches

The following optimization techniques were implemented and evaluated in this project:

  - **Baseline (`None`):** The standard, unoptimized model serving as a control for all comparisons. It uses full 32-bit floating-point (FP32) precision for all computations.

  - **FP16 (Mixed Precision):** This technique uses 16-bit floating-point numbers for certain operations where precision is not critical, while keeping model weights in FP32. This significantly reduces VRAM usage and can speed up computations on GPUs with Tensor Cores.

  - **Pruning:** A technique to reduce the model size by removing weights that are close to zero. The implemented method is L1 unstructured pruning, which removes a percentage of the least important weights across all layers.

  - **Dynamic Quantization:** This method converts model weights from 32-bit floats to 8-bit integers during runtime. It is primarily used to reduce model size and accelerate inference on CPUs.

  - **Scripting:** This involves using `torch.jit.script` to analyze and compile the model's graph into a static representation. This can enable a range of low-level optimizations not possible with standard eager-mode execution.

## 5\. Results Summary with Key Insights

The final `benchmark_results.csv` file provides a comprehensive dataset of all benchmark runs. Key insights from the data include:

  - **FP16 is the clear winner:** It provided the most significant improvements in both VRAM usage and throughput.
  - **Pruning was unsuccessful:** The model size surprisingly increased after the pruning operation, indicating a potential issue with the implementation.
  - **Scripting had minimal effect:** This optimization did not provide any noticeable performance or memory benefits for this particular model.

## 6\. Performance Analysis

A detailed analysis of the collected data reveals the following:

  - **Baseline Performance:** As expected, the baseline model's VRAM usage scaled directly with batch size. The throughput initially increased with batch size but showed a significant drop at a batch size of 16, indicating a potential bottleneck or resource saturation.

  - **FP16 Performance:** The `fp16` optimization achieved the highest throughput at a batch size of 16, reaching `42.18` samples/sec. More importantly, it reduced VRAM usage by over 40%, dropping from `4065.44` MB to just `2407.44` MB at a batch size of 16.

  - **Pruning Performance:** The `pruning` technique failed to reduce the model size, which unexpectedly grew from `30.75` MB to `56.96` MB. Performance metrics remained comparable to the baseline, indicating that the intended optimization was not achieved.

  - **Scripting Performance:** The `script` optimization did not result in a significant change in performance, with all metrics remaining very close to the baseline. This suggests that the model's graph was already simple enough that further compilation provided no additional benefit.

## 7\. Trade-offs Discussion

The primary trade-off observed was a direct benefit from using `fp16` with no apparent drawbacks. The other optimizations did not present a significant trade-off as they failed to provide the expected benefits.

  - **FP16:** Provided a clear win in both VRAM and throughput, demonstrating an excellent performance-to-cost ratio.
  - **Pruning:** The implementation had a major trade-off of increased model size, making it a poor choice for production.
  - **Dynamic Quantization:** While a powerful technique, it is primarily optimized for CPU inference, which may explain its minimal impact on GPU metrics.

## 8\. Known Limitations

  - **Accuracy Not Measured:** This benchmark focuses solely on performance and does not include a validation dataset to measure the impact of optimizations on model accuracy.
  - **Incomplete Pruning:** The current pruning implementation is flawed, as it did not reduce the model size.
  - **Limited Hardware:** Results are specific to the hardware used in the benchmarking environment and may vary on different GPUs.

## 9\. Future Improvements

  - **Implement a Better Pruning Technique:** Use a more robust pruning method and ensure the model size is correctly reduced.
  - **Measure Accuracy:** Add a validation loop to the benchmark script to collect `accuracy_top1` and `accuracy_top5` metrics.
  - **Expand Model Support:** Add support for other models (e.g., ResNet, VGG) and compare optimization performance across different architectures.
  - **Include More Optimizations:** Integrate other techniques like ONNX or TensorRT for a more comprehensive comparison.

-----

### Code Documentation

In addition to the `README.md`, the Python codebase itself is documented to ensure clarity and maintainability.

  - **Docstrings:** All major functions include comprehensive docstrings explaining their purpose, arguments, and return values.
  - **Type Hints:** Type hints are used throughout the codebase to improve readability and enable static analysis.
  - **Inline Comments:** Complex logic, particularly within the benchmarking loop and optimization blocks, is explained with inline comments.