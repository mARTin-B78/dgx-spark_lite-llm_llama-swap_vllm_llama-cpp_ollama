FROM spark-vllm:Version_1
RUN pip install causal-conv1d mamba_ssm --no-cache-dir
RUN pip install vllm[audio]==0.20.0 torch nvidia-cutlass-dsl flashinfer-cubin==0.6.8.post1 flashinfer-python==0.6.8.post1 --no-cache-dir \
 && pip uninstall -y flashinfer-jit-cache
ENV FLASHINFER_DISABLE_VERSION_CHECK=1