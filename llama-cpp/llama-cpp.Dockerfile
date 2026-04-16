# lama-cpp.Dockerfile
# Must use 13.1 so the compiler understands architecture "121"
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS builder

# 1. Install prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Fix the Sbsa ARM64 stubs by creating the missing .1 link inside the NVIDIA targets folder
RUN cd /usr/local/cuda/targets/sbsa-linux/lib/stubs && ln -sf libcuda.so libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib/stubs:$LIBRARY_PATH

WORKDIR /app
RUN git clone https://github.com/ggml-org/llama.cpp src

# 3. Build for Spark GB10 (Architecture 121)
RUN cd src && mkdir build && cd build && \
    cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="121" \
    -DLLAMA_CURL=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link=/usr/local/cuda/targets/sbsa-linux/lib/stubs" \
    && make -j$(nproc)

# 4. Clean Runtime Stage
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

# --- THE FIX --- 
# Copy the compiled binary AND all the required shared libraries (*.so)
COPY --from=builder /app/src/build/bin/llama-server /app/llama-server
COPY --from=builder /app/src/build/bin/*.so* /usr/lib/

# Expose the new port
EXPOSE 18080

ENTRYPOINT ["/app/llama-server"]
