# docker build -t smoldocling-lambda --network sagemaker -f lambda-custom_runtime.Dockerfile  . 
# Stage 1: Builder image with full toolchain
FROM --platform=linux/amd64 public.ecr.aws/lambda/provided:al2023 as builder

# Install essential build tools
RUN dnf install -y \
    git \
    cmake \
    make \
    gcc-c++ \
    openssl-devel \
    libgomp \
    libcurl-devel  \
    zlib-devel

# Clone and build llama.cpp with AVX2 optimizations
RUN mkdir llama.cpp && \
    cd llama.cpp && \
    git init && \
    git remote add origin https://gitea.swigg.net/dustins/llama.cpp.git && \
    git fetch origin c3a654c0fbad4c7eeeaf669fc708d40aef6f341c && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF  && \
    cmake --build build --config Release -t llama-server



FROM --platform=linux/amd64 public.ecr.aws/lambda/provided:al2023


ENV LAMBDA_TASK_ROOT=/var/task
ENV LAMBDA_RUNTIME_DIR=/var/runtime

RUN dnf install -y libgomp jq && \
    dnf clean all


# Set critical environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Copy built artifacts
COPY --from=builder ${LAMBDA_TASK_ROOT}/llama.cpp/build/bin/llama-server ${LAMBDA_TASK_ROOT}/server

# Add model files (replace with your GGUF model)
COPY smoldocling-256M.fp16.gguf  /var/task/models/

# Configuration
ENV MODEL_PATH=/var/task/models/smoldocling-256M.fp16.gguf 


COPY bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap
RUN chmod 755 ${LAMBDA_RUNTIME_DIR}/bootstrap

COPY function.sh ${LAMBDA_TASK_ROOT}/function.sh
RUN chmod 755 ${LAMBDA_TASK_ROOT}/function.sh

CMD [ "function.handler" ]
# Set bootstrap as entrypoint
# ENTRYPOINT [${LAMBDA_RUNTIME_DIR}/bootstrap]



# curl -v -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{"prompt":"<|im_start|>system You are a helpful assistant<|im_end|> <|im_start|>user Hello<|im_end|> <|im_start|>assistant Hi there<|im_end|> <|im_start|>user How are you?<|im_end|> <|im_start|>assistant","n_predict":128,"temperature":0.2}'



# curl -v -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{ "path":"/v1/chat/completions", "method":"POST", "body":{ "model": "/var/task/models/smoldocling-256M.fp16.gguf", "messages": [  {"role": "user", "content": "I believe the meaning of life is"} ], "temperature": 0.0, "generation_settings": {"n_ctx":4096} }}'

# curl -v -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{ "path":"/health"}'

# curl -v -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{ "path":"/v1/completions", "method":"POST", "body":{ "prompt":"<|im_start|>system You are a helpful assistant<|im_end|> <|im_start|>user Hello<|im_end|> <|im_start|>assistant Hi there<|im_end|> <|im_start|>user How are you?<|im_end|> <|im_start|>assistant", "temperature": 0.2 }}'

# curl -v -XPOST "http://localhost:8080/2015-03-31/functions/function/invocations" -d '{ "path":"/v1/models"}'


# docker run -v ~/.aws-lambda-rie:/aws-lambda --network sagemaker   -e AWS_LAMBDA_FUNCTION_NAME="SmolDoclingFunction"   --entrypoint /aws-lambda/aws-lambda-rie  smoldocling-lambda /var/runtime/bootstrap function.handler


#../llama.cpp/build/bin/llama-cli -m /home/sagemaker-user/sample-for-multi-modal-document-to-json-with-sagemaker-ai/docker-artifacts/smoldocling-256M.fp16.gguf -p "I believe the meaning of life is" -n 128 -no-cnv --temp 0.0


# ../llama.cpp/build/bin/llama-vision -m /home/sagemaker-user/sample-for-multi-modal-document-to-json-with-sagemaker-ai/docker-artifacts/smoldocling-256M.fp16.gguf -p "I believe the meaning of life is" -n 128 --temp 0.1 --image /home/sagemaker-user/sample-for-multi-modal-document-to-json-with-sagemaker-ai/imag.jpg


