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
    curl-devel \
    zlib-devel

# Build AWS Lambda C++ Runtime
RUN git clone https://github.com/awslabs/aws-lambda-cpp.git && \
    cd aws-lambda-cpp && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && \
    make -j$(nproc) && make install

# Clone and build llama.cpp with AVX2 optimizations
RUN mkdir llama.cpp && \
    cd llama.cpp && \
    git init && \
    git remote add origin https://gitea.swigg.net/dustins/llama.cpp.git && \
    git fetch origin c3a654c0fbad4c7eeeaf669fc708d40aef6f341c && \
    git checkout FETCH_HEAD && \
    mkdir build && \
    cmake -B build -DLLAMA_AVX2=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=OFF -DLLAMA_BUILD_SERVER=ON -DBUILD_SHARED_LIBS=OFF  && \
    cmake --build build --config Release -t llama-server


# Stage 2: Final runtime image
FROM --platform=linux/amd64 public.ecr.aws/lambda/provided:al2023


RUN dnf install -y libgomp && \
    dnf clean all


# Set critical environment variables
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH





# Copy built artifacts
COPY --from=builder /var/task/llama.cpp/build/bin/llama-server /var/task/server
COPY --from=builder /var/task/aws-lambda-cpp/build/libaws-lambda-runtime.a /var/runtime/

# Add model files (replace with your GGUF model)
COPY smoldocling-256M.fp16.gguf  /var/task/models/

# Configuration
ENV MODEL_PATH=/var/task/models/smoldocling-256M.fp16.gguf 

# COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.0 /lambda-adapter /opt/extensions/lambda-adapter
# RUN chmod +x /opt/extensions/lambda-adapter

# Set critical environment variables
# Standard Lambda port (from search result 3)
#ENV PORT=8080 
ENV WEB_ADAPTER_LOG_LEVEL=DEBUG

# Install runtime dependencies
# RUN dnf install -y curl jq && dnf clean all

# Copy and set up entrypoint script
COPY lambda-entrypoint.sh /var/runtime/bootstrap
RUN chmod +x /var/runtime/bootstrap

# Set bootstrap as entrypoint
ENTRYPOINT ["/var/runtime/bootstrap"]
