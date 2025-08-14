
## Overview

The setup includes four different service configurations:
1. **Basic Service** - Single request processing
2. **Batched Service** - Multiple request batching for improved throughput
3. **Compiled Service** - PyTorch compilation for faster inference
4. **Compiled + Batched Service** - Combined optimizations

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

## Setup

### 1. Initial Environment Setup

```bash
# Clone the repository and navigate to the DiT serving directory
cd torchserve_serving/serve_dit

# Install dependencies and download the model
chmod +x setup_env.sh
./setup_env.sh
```

This script will:

1. Install required Python packages
2. Install TorchServe and dependencies
3. Download the DiT-XL-2-256 model from Hugging Face

### 2. Service Configurations
#### 1. Basic Service (Single Request)

Configuration: config.properties

- Port: 8080 (inference), 8081 (management), 8082 (metrics)
- Batch size: 1
- Max workers: 1

```bash
./start_service.sh
```

#### 2. Batched Service
Configuration: config_batched.properties

- Port: 8080 (inference), 8081 (management), 8082 (metrics)
- Batch size: 4
- Max batch delay: 100ms
- Max workers: 1

```bash
./start_batched_service.sh
```