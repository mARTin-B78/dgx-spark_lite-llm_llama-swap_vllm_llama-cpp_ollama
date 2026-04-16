# lama-swap.Dockerfile

FROM ubuntu:22.04

# Install dependencies AND docker.io so llama-swap can manage containers
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ca-certificates \
    wget \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Download the latest ARM64 release of llama-swap
RUN mkdir -p /tmp/llama-swap-build && \
    cd /tmp/llama-swap-build && \
    curl -sL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | \
    grep "browser_download_url.*linux_arm64.tar.gz" | \
    head -1 | \
    cut -d'"' -f4 | \
    xargs wget -q && \
    tar -xzf llama-swap_*linux_arm64.tar.gz && \
    mv llama-swap /usr/bin/llama-swap && \
    chmod +x /usr/bin/llama-swap && \
    cd / && \
    rm -rf /tmp/llama-swap-build

WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["/usr/bin/llama-swap"]
