# DiTServing
## Prerequisites
- **Python**: 3.8 – 3.11 (3.10 recommended)
- **NVIDIA GPU + CUDA**: 11.8+ or 12.x (must match PyTorch version)
- **cuDNN**: Compatible version with your CUDA installation

---

## Getting Started

### 1. Installation
```bash
cd glideserve
pip install -e .
```

### 2. Download Models
```bash
cd scripts
./download_sd3_from_links.sh
```

### 3. Usage
#### Start the Server
```bash
cd glideserve
python server.py
```
#### Single and Batched Inference
```bash
cd scripts
# Single prompt
python run_simple_example.py "A beautiful landscape" --timesteps 50

# Batched prompts
python run_batched_example.py test_prompts.json
```

## Benchmarking
Benchmarks were run on an **NVIDIA A100 GPU**.  
_For detailed usage instructions on running these benchmarks, see the [Benchmarking Guide](./benchmarking/sd3_glideserve_benchmarking/benchmarking.md)._

#### Results
| Server Tested           | Mean Time (s) | Median Time (s) | Min Time (s) | Max Time (s) |
|-------------------------|---------------|-----------------|--------------|--------------|
| GlideServe (Single)     | 1.23          | 1.10            | 1.35         |              | 
| GlideServe (Batched)    | 0.85          | 0.80            | 0.92         |              |


# Load Testing
## Results
