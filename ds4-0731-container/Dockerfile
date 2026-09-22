# Runtime reuses a locally available CUDA 13 / Ubuntu 24.04 base.
ARG BASE_IMAGE=local/ds4-runtime-base:cd5b279db639
FROM ${BASE_IMAGE}
COPY ds4-server /opt/ds4/ds4-server
WORKDIR /opt/ds4
ENTRYPOINT ["/opt/ds4/ds4-server"]
