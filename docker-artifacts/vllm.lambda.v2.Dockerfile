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
    && curl -fsSL https://pixi.sh/install.sh | sh


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


ENV PATH="/root/.pixi/bin:${PATH}"
RUN pixi init --format pyproject
ENV UV_HTTP_TIMEOUT=500

# Install Python dependencies 
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"
RUN pixi add "python==3.12" && \
    pixi run uv pip install -r vllm_source/requirements/build.txt && \
    pixi run uv pip install -r vllm_source/requirements/cpu.txt && \
    pixi add --pypi py-cpuinfo  # Use this to gather CPU info and optimize based on ARM Neoverse cores


ENV LD_PRELOAD="/usr/lib64/libtcmalloc_minimal.so.4:$LD_PRELOAD"


RUN echo 'ulimit -c 0' >> ~/.bashrc

# Disabling AVX512 specific optimizations for ARM
ARG VLLM_CPU_DISABLE_AVX512="true"
ENV VLLM_CPU_DISABLE_AVX512=${VLLM_CPU_DISABLE_AVX512}

# Build vLLM with CPU support
WORKDIR vllm_source

RUN VLLM_TARGET_DEVICE=cpu pixi run python setup.py bdist_wheel && \
    pixi add --pypi dist/*.whl && \
    rm -rf dist