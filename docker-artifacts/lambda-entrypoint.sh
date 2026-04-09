#!/bin/bash
set -euo pipefail

# Start llama.cpp server in background
echo "Starting llama.cpp server..."
/var/task/server \
  -m $MODEL_PATH \
  --host 0.0.0.0 \
  --port 9001 \
  -c 4096 \
  -t 4 \
  --n-predict 512 &

SERVER_PID=$!

# Wait for server to initialize
sleep 3
echo "Server started with PID: $SERVER_PID"


# Processing loop to handle Lambda invocations
while true
do
  echo "Waiting for invocation..."
  HEADERS="$(mktemp)"
  
  # Get next invocation from Lambda Runtime API
  EVENT_DATA=$(curl -sS -LD "$HEADERS" "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/next")
  
  # Extract request ID from headers
  REQUEST_ID=$(grep -Fi Lambda-Runtime-Aws-Request-Id "$HEADERS" | tr -d '[:space:]' | cut -d: -f2)
  echo "Processing request: $REQUEST_ID"
  
  # Parse event data
  PROMPT=$(echo $EVENT_DATA | jq -r '.prompt // .body // "{\"prompt\": \"Hello\", \"max_tokens\": 128}"')
  
  # If prompt is still json string (from body field), parse it again
  if [[ $PROMPT == {* ]]; then
    PROMPT=$(echo $PROMPT | jq -r '.prompt // "Hello"')
  fi
  
  # Forward request to llama.cpp server
  RESPONSE=$(curl -s -X POST "http://localhost:9001/completion" \
             -H "Content-Type: application/json" \
             -d "{\"prompt\":\"$PROMPT\",\"n_predict\":128}")
  
  # Send response back to Lambda
  echo "Sending response for request: $REQUEST_ID"
  curl -s -X POST "http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/$REQUEST_ID/response" \
       -d "$RESPONSE"
done


# docker run -v ~/.aws-lambda-rie:/aws-lambda --network sagemaker -e AWS_LAMBDA_RUNTIME_API=127.0.0.1:8080 \
#   -e AWS_LAMBDA_FUNCTION_NAME="SmolDoclingFunction" smoldocling-lambda:latest  \
#     --entrypoint /aws-lambda/aws-lambda-rie  /var/runtime/bootstrap 
    
# docker run -v ~/.aws-lambda-rie:/aws-lambda -e AWS_LAMBDA_RUNTIME_API="127.0.0.1:9001" --network sagemaker smoldocling-lambda:latest  \
#     --entrypoint /aws-lambda/aws-lambda-rie /bin/bash


# docker run -it --rm \
#   -v ~/.aws-lambda-rie:/aws-lambda \
#   -e AWS_LAMBDA_RUNTIME_API="127.0.0.1:9001" \
#   --network sagemaker \
#   --entrypoint /bin/bash \
#   smoldocling-lambda:latest



# curl -s -X POST "http://localhost:9001/completion" \
#              -H "Content-Type: application/json" \
#              -d "{\"prompt\":\"Hello\",\"n_predict\":128}"
