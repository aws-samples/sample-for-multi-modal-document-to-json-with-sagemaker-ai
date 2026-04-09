# Project Structure

## Root Directory
```
├── 01_optional_convert_fatura2_to_hf_dataset.ipynb  # Dataset conversion to HF format
├── 02_create_custom_dataset_*.ipynb                 # Training data preparation (SWIFT/Nova)
├── 03_finetune_*.ipynb                              # Model fine-tuning notebooks
├── 04_*.ipynb                                       # Batch inference / deployment
├── 05_evaluate_model.ipynb                          # Model evaluation
├── 06_deploy_model_endpoint.ipynb                   # Endpoint deployment
├── 07_consume_model.ipynb                           # Endpoint invocation
├── 08_lambda_deployment.ipynb                       # Lambda deployment
├── config.yaml                                      # SageMaker configuration
└── README.md                                        # Project documentation
```

## Key Directories

### `/utils/` - Shared Python Utilities
- `helpers.py` - S3 operations, SageMaker job management, path utilities
- `evaluation.py` - JSON parsing, response processing, metrics extraction
- `finetuning.py` - Checkpoint management, training job utilities
- `bedrock.py` - Bedrock API helpers, training metrics visualization
- `dist.py` - Edit distance calculations
- `docdiff.py` - Visual diff for document comparison
- `entities.py` - Entity analysis and categorization
- `config.py` - Dataclass configurations for models

### `/data/` - Datasets and Results
- `Fatura2-invoices-original-strat2/` - Invoice dataset with images and JSONL annotations
- `fake-w2-us-tax-form-dataset/` - W2 tax form dataset
- `processed/` - Prepared training data in Nova/SWIFT format
- `results/` - Inference outputs and evaluation results

### `/docker-artifacts/` - Container Definitions
- Dockerfiles for SageMaker and Lambda deployment
- Bootstrap and entrypoint scripts for VLLM inference

### `/images/` - Evaluation Visualizations
- Heatmaps and charts from model evaluation

### `/models/` - Local Model Files
- Downloaded model weights (e.g., smoldocling-256M)

## Naming Conventions
- Notebooks: Numbered prefix (01_, 02_) indicates execution order
- Training jobs: `finetune-{model}-{dataset}-{timestamp}`
- Checkpoints: `v{version}-{YYYYMMDD}-{HHMMSS}/checkpoint-{step}`
