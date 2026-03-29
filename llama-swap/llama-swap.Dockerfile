FROM golang:1.22-bookworm AS build
WORKDIR /app
RUN git clone https://github.com/mostlygeek/llama-swap .
RUN go build -o llama-swap ./cmd/llama-swap

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y curl docker.io && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/llama-swap /app/llama-swap
ENTRYPOINT ["/app/llama-swap"]
