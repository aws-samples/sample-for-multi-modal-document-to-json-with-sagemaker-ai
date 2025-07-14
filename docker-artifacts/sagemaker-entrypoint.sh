#!/bin/bash
set -x
# Define the prefix for environment variables to look for
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
echo "Starting server on port $PORT"
ARGS=(--port $PORT)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Download the model
#MODEL_ID_URI=${ADAPTER_URI}

# Construct the environment variable name
ENV_VAR_NAME="ADAPTER_URI"

# Get the value from the environment variable
MODEL_URI="${!ENV_VAR_NAME}"

# Check if the environment variable exists and is not empty
if [ -z "$MODEL_URI" ]; then
    echo "Error: Environment variable ${ENV_VAR_NAME} is not set or empty"
    exit 1
fi


# Download the model
MODEL_DIR=/opt/ml/model
mkdir -p $MODEL_DIR

export USE_HF_TRANSFER=1
# export PATH=$PATH:$(python3 -m site --user-base)/bin
/usr/local/bin/aws s3 cp "$MODEL_URI" "$MODEL_DIR"

tar -xf "${MODEL_DIR}/model.tar.gz" -C "$MODEL_DIR"

# Get the best model from potentially multiple checkpoints:
# Sample: /opt/ml/checkpoints/v0-20250214-124042/checkpoint-500
BEST_MODEL_CHECKPOINT=$(find /opt/ml/model -name logging.jsonl -exec grep -hoP '"best_model_checkpoint":\s*"\K[^"]*' {} \; | head -n1)
BEST_MODEL_DIR=${BEST_MODEL_CHECKPOINT#/opt/ml/checkpoints/}


GPU_MEMORY_UTILIZATION=0.95

# First, merge the QLoRA/LoRA adapter with the base model
# Note vllm only supports LoRA merging, not QLoRA merging. Which is why we use swift to merge.
echo "====> Merging QLoRA/LoRA adapter with base model"
swift export --adapters $BEST_MODEL_DIR --merge_lora true --use_hf true
echo "====> QLoRA/LoRA merging completed"

# Check if merged model directory exists
MERGED_MODEL_DIR="${BEST_MODEL_DIR}-merged"
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "Error: Merged model directory not found at $MERGED_MODEL_DIR"
    echo "Contents of ${BEST_MODEL_DIR}:"
    ls -la ${BEST_MODEL_DIR}
    echo "Contents of parent directory:"
    ls -la $(dirname ${BEST_MODEL_DIR})
    exit 1
fi

# Detect number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "====> Detected $NUM_GPUS GPUs available"
else
    NUM_GPUS=1
    echo "====> No GPUs detected, defaulting to CPU mode"
fi

echo "====> Starting vllm serve with merged model: $MERGED_MODEL_DIR"
# Now serve the model with vllm
exec vllm serve \
  "$MERGED_MODEL_DIR" \
  --tokenizer "$MERGED_MODEL_DIR" \
  --no-skip-tokenizer-init \
  --port 8080 \
  --host 0.0.0.0 \
  --tokenizer-mode auto \
  --trust-remote-code \
  --allowed-local-media-path "" \
  --load-format auto \
  --kv-cache-dtype auto \
  --seed 0 \
  --pipeline-parallel-size 1 \
  --tensor-parallel-size $NUM_GPUS \
  --swap-space 4 \
  --cpu-offload-gb 0 \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-partial-prefills 1 \
  --max-long-partial-prefills 1 \
  --long-prefill-token-threshold 0 \
  --max-logprobs 20 \
  --disable-log-stats \
  --max-seq-len-to-capture 8192 \
  --guided-decoding-backend xgrammar \
  --disable-async-output-proc \
  --scheduling-policy fcfs \
  --scheduler-cls vllm.core.scheduler.Scheduler \
  --worker-cls auto \
  --model-impl auto \
  --disable-log-requests \
  "${ARGS[@]}"

# exec swift deploy --adapters $BEST_MODEL_DIR --merge_lora true --load_data_args true --infer_backend vllm --max_num_seqs $MAX_NUM_SEQS --gpu_memory_utilization $GPU_MEMORY_UTILIZATION --max_model_len $MAX_MODEL_LEN\
#         --use_hf true \
#         "${ARGS[@]}"

