# Modified vLLM Dockerfile for AWS Lambda with CPU support
# 
# Build arguments:
#   PYTHON_VERSION=3.12 (default)
#   VLLM_CPU_DISABLE_AVX512=false (default)|true
#
ARG VIRTUAL_ENV="/opt/venv"

######################### BUILD IMAGE #########################
FROM --platform=linux/amd64 public.ecr.aws/lambda/provided:al2023 as builder
ARG VIRTUAL_ENV
ENV BUILD_DIR "/build"
WORKDIR $BUILD_DIR

ARG PYTHON_VERSION=3.12
ARG PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"

# Install essential build tools

#ccache git curl wget ca-certificates \
 #       gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 \
  #  && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
   # && curl -LsSf https://astral.sh/uv/install.sh | sh

######################### Install minimal dependencies and uv #########################
RUN dnf install -y \
    git \
    wget \
    ca-certificates \
    gperftools \
    cmake \
    make \
    gcc \
    gcc-c++ \
    openssl-devel \
    libgomp \
    zlib-devel \
    tar \
    numactl-devel \
    autoconf \
    automake \
    bzip2 \
    bzip2-devel \
    freetype-devel \
    libtool \
    pkgconfig \
    glibc-static \
    zlib-devel \
    yasm nasm \
    && dnf clean all \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

######################### Install ffmpeg #########################

ENV FFMPEG_VERSION=4.2.2

ARG PREFIX=/opt/ffmpeg

## ffmpeg https://ffmpeg.org/

ENV PKG_CONFIG_PATH=/usr/lib64/pkgconfig:$PKG_CONFIG_PATH


RUN DIR=/tmp/ffmpeg && mkdir -p ${DIR} && cd ${DIR} && \
    curl -O -L https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
    ls && \
    tar xjvf ffmpeg-snapshot.tar.bz2 

ENV PATH="$PREFIX/bin:$PATH"

RUN DIR=/tmp/ffmpeg && mkdir -p ${DIR} && cd ${DIR}/ffmpeg && \
    ./configure \
    --prefix="${PREFIX}" \
    --bindir="${PREFIX}/bin" \
    --extra-cflags="-I${PREFIX}/include -fstack-protector-strong -fpie -pie -Wl,-z,relro,-z,now -D_FORTIFY_SOURCE=2" \
    --extra-ldflags="-L${PREFIX}/lib" \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --extra-libs=-lpthread \
    --extra-libs=-lm \
    --enable-libfreetype \
    --disable-static \
    --enable-shared \
    --enable-rpath  && \
    make && \
    make install

######################### Install VLLM with CPU Support #########################
RUN git clone https://github.com/vllm-project/vllm.git vllm_source

#ENV CCACHE_DIR=/root/.cache/ccache
#ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV=$VIRTUAL_ENV
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV UV_HTTP_TIMEOUT=500

# Install Python dependencies 
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"
RUN uv pip install --upgrade pip && \ 
    uv pip install -r vllm_source/requirements/build.txt && \
    uv pip install -r vllm_source/requirements/cpu.txt && \
    uv pip install py-cpuinfo  # Use this to gather CPU info and optimize based on ARM Neoverse cores

# RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install intel-openmp==2024.2.1 intel_extension_for_pytorch==2.6.0

ENV LD_PRELOAD="/usr/lib64/libtcmalloc_minimal.so.4:$LD_PRELOAD"


RUN echo 'ulimit -c 0' >> ~/.bashrc

# Disabling AVX512 specific optimizations for ARM
ARG VLLM_CPU_DISABLE_AVX512="true"
ENV VLLM_CPU_DISABLE_AVX512=${VLLM_CPU_DISABLE_AVX512}

# Build vLLM with CPU support
WORKDIR vllm_source

RUN VLLM_TARGET_DEVICE=cpu python3 setup.py bdist_wheel && \
    uv pip install dist/*.whl && \
    rm -rf dist


######################### RUNTIME IMAGE #########################
FROM --platform=linux/amd64 public.ecr.aws/lambda/provided:al2023
ARG VIRTUAL_ENV
ENV VIRTUAL_ENV=$VIRTUAL_ENV

ENV LAMBDA_TASK_ROOT=/var/task
ENV LAMBDA_RUNTIME_DIR=/var/runtime

# Set environment variables
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV WEB_ADAPTER_LOG_LEVEL=DEBUG

ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV LD_PRELOAD="/usr/lib64/libtcmalloc_minimal.so.4:$LD_PRELOAD"

ENV VLLM_CPU_KVCACHE_SPACE=40


# Install minimal runtime dependencies
RUN dnf install -y \
    numactl \
    libgomp \
    jq \
    python3.12 \
    gperftools \
    && dnf clean all



# Copy the Python virtual environment with vLLM installed
COPY --from=builder ${VIRTUAL_ENV} ${LAMBDA_TASK_ROOT}/vllm

# Copy and set up entrypoint script
COPY bootstrap.vllm.sh ${LAMBDA_RUNTIME_DIR}/bootstrap
RUN chmod 755 ${LAMBDA_RUNTIME_DIR}/bootstrap

COPY function.vllm.sh ${LAMBDA_TASK_ROOT}/function.sh
RUN chmod 755 ${LAMBDA_TASK_ROOT}/function.sh



WORKDIR ${LAMBDA_TASK_ROOT}/vllm/lib/python3.12/site-packages


CMD [ "function.handler" ]
# Set bootstrap as entrypoint
#ENTRYPOINT ["/var/runtime/bootstrap"]