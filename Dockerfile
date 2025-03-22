FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers.git@26149c0ecda67587ffd51f1a91c888388f83253b

# Copy model files
COPY predict.py .
COPY replicate_cog_wrapper.py .

# Set environment variables for Replicate
ENV PYTHONPATH=/app
ENV REPLICATE_INPUT_PATH=/tmp/input.json
ENV REPLICATE_OUTPUT_PATH=/tmp/output.json

# Create directory for outputs
RUN mkdir -p /app/outputs

# Run the model
ENTRYPOINT ["python3", "replicate_cog_wrapper.py", "predict"]