### `README.md`

# Deep Learning Model Optimization Benchmark

### 1\. Overview

This project benchmarks the performance of a pre-trained `DenseNet-121` deep learning model under various optimization techniques. The goal is to measure and compare key metrics such as latency, throughput, and memory consumption to determine the most effective optimization for this specific model on a GPU-enabled environment.

The following optimizations were benchmarked:

  * **Baseline:** The standard, unoptimized model.
  * **FP16:** Mixed-precision inference using 16-bit floating-point numbers.
  * **Pruning:** Reducing the model size by removing weights.
  * **Dynamic Quantization:** Converting model weights to 8-bit integers on the fly.
  * **Scripting:** Compiling the model using TorchScript to enable graph-level optimizations.

### 2\. Project Structure

The project is containerized using Docker to ensure a consistent and reproducible environment.

  - `src/main.py`: The core Python script that loads the model, applies each optimization, runs the benchmarks, and logs the results.
  - `docker-compose.yml`: Defines the Docker services and dependencies, including the `benchmark-app` service that runs the main script.
  - `build_and_run.sh`: A convenience script to build the Docker image and run the benchmark in a single command.
  - `results/benchmark_results.csv`: The final output file containing all the collected benchmark data.

### 3\. How to Run

To replicate the benchmark, navigate to the project's root directory in your terminal and execute the `build_and_run.sh` script.

```bash
./build_and_run.sh
```

This script will build the Docker image and run the benchmark. The process will take a few minutes as it measures each optimization across multiple batch sizes. The final results will be saved in `results/benchmark_results.csv`.


### 4\. Analysis and Conclusions

#### Baseline (`none` and `densenet121_baseline`)

The baseline runs show that as the batch size increases, VRAM usage also increases linearly. Throughput generally increases with batch size, but drops significantly at batch size 16. This is a common occurrence on some GPUs as larger batches can saturate the memory and compute resources, leading to reduced efficiency.

#### `FP16` (Mixed Precision)

The `fp16` optimization was highly effective.

  - **VRAM Reduction:** This technique dramatically reduced VRAM usage for all batch sizes. For a batch size of 16, VRAM dropped from `4065.44` MB (baseline) to `2407.44` MB (fp16), representing a significant memory saving.
  - **Throughput:** `fp16` provided the highest throughput at a batch size of 16, reaching `42.18` samples per second. It also showed improvements for other batch sizes compared to the baseline.

#### `Pruning`

The pruning optimization did not perform as expected.

  - **Model Size Increase:** Unexpectedly, the model size increased to `56.96` MB after pruning, compared to the baseline size of `30.75` MB. This suggests an issue with the implementation or the measurement method.
  - **Performance:** Despite the model size increase, the throughput and latency metrics were comparable to the baseline, which is not the expected outcome of a pruning operation.

#### `Scripting`

Scripting had a minimal effect on the benchmark metrics.

  - **Performance & VRAM:** The throughput, latency, and VRAM usage metrics for the `script` optimization were very similar to the baseline, showing no clear benefit from this technique for this model on the test hardware.
  - **Model Size:** The model size remained unchanged at `30.75` MB.

#### Final Conclusion

Based on these results, **FP16 mixed precision** is the most effective optimization for the `DenseNet-121` model in this benchmarking environment. It provided a clear benefit in VRAM reduction and delivered the highest throughput at the largest batch size, making it the best choice for improving inference performance. The `pruning` and `script` optimizations did not yield the expected results and would require further investigation to be considered a viable solution.