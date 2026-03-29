FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS build
WORKDIR /app
RUN apt-get update && apt-get install -y git cmake build-essential libcurl4-openssl-dev
RUN git clone https://github.com/ggml-org/llama.cpp .
RUN cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real && \
    cmake --build build --config Release -j$(nproc)

FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04
COPY --from=build /app/build/bin/llama-server /app/llama-server
ENTRYPOINT ["/app/llama-server"]
