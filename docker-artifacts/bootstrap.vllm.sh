#!/bin/sh

set -euo pipefail


export USE_HF_TRANSFER=1

# Check if the environment variable exists and is not empty
if [ -z "$MODEL_ID" ]; then
    echo "Error: Environment variable ${ENV_VAR_NAME} is not set or empty"
    exit 1
fi

echo "Downloading model..."
# TODO

# Start vllm server in background
echo "Starting vllm server..."

exec python3.12 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --port 9002 \
    --host 0.0.0.0 \
    --trust-remote-code &
  # Todo add all the other parameter
  

# Initialization - load function handler
source $LAMBDA_TASK_ROOT/"$(echo $_HANDLER | cut -d. -f1).sh"

# Processing
while true
do
  HEADERS="$(mktemp)"
  # Get an event. The HTTP request will block until one is received
  EVENT_DATA=$(curl -sS -LD "$HEADERS" -X GET "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next")

  # Extract request ID by scraping response headers received above
  REQUEST_ID=$(grep -Fi Lambda-Runtime-Aws-Request-Id "$HEADERS" | tr -d '[:space:]' | cut -d: -f2)

  # Forward request to llama.cpp server
   #RESPONSE=$(curl -s -X POST "http://localhost:9002/v1/chat/completions" \
  #            -H "Content-Type: application/json" \
    #          -d "$EVENT_DATA")



  # Run the handler function from the script
  # RESPONSE=$($(echo "$_HANDLER" | cut -d. -f2) "$EVENT_DATA")

  RESPONSE=$($(echo "$_HANDLER" | cut -d. -f2) "$EVENT_DATA") || {
    curl -s -X POST "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/$REQUEST_ID/error" \
      -d '{"error":"Invalid endpoint","type":"InvalidRoute"}'
    continue
  }

  # Send the response
  curl -X POST "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/$REQUEST_ID/response"  -d "$RESPONSE"
done