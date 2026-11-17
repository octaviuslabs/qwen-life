FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /workspace/qwen-life

RUN python -m pip install --no-cache-dir \
    "transformers==5.22.1" \
    "safetensors>=0.8.0" \
    "tokenizers>=0.22.0" \
    "pytest>=8.0"

ENV PYTHONPATH=/workspace/qwen-life:/workspace/qwen-life/references/qwenputer:/workspace/qwen-life/references/qwenGPU

