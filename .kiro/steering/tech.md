# Tech Stack

## Core Technologies
- Python 3.x (Jupyter notebooks)
- Amazon SageMaker AI (training, batch inference, endpoints)
- Amazon Bedrock (Nova model fine-tuning)
- AWS S3 (data storage)

## Key Libraries
- `boto3`, `sagemaker` - AWS SDK and SageMaker Python SDK
- `pandas` - Data manipulation
- `Pillow (PIL)` - Image processing
- `tqdm` - Progress bars
- `Levenshtein` - Edit distance calculations
- `regex` - Advanced regex for JSON parsing
- `matplotlib`, `seaborn` - Visualization
- `difflib` - Text comparison

## ML Frameworks
- ModelScope SWIFT - Fine-tuning framework for VLMs
- VLLM - Inference with structured output/JSON constrained decoding
- Hugging Face - Model hub and datasets

## Configuration
- `config.yaml` - SageMaker remote function settings (instance types, S3 paths, environment variables)

## Common Commands

### Install Dependencies
```bash
pip install boto3 sagemaker pandas Pillow tqdm python-Levenshtein regex matplotlib seaborn
```

### Run Notebooks
Execute notebooks sequentially (01 → 07) in SageMaker Studio or local Jupyter environment with AWS credentials configured.

### Docker (for deployment)
```bash
# Build and push container for endpoint deployment
cd docker-artifacts
./02_build_and_push.sh
```

## AWS Regions
- `us-west-2` - Default for SageMaker training
- `us-east-1` - Required for Amazon Nova/Bedrock fine-tuning
