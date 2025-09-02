# Face Enhancer Docker Image - RunPod Serverless
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg wget curl git unzip \
    build-essential python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgoogle-perftools4 libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Create directories for models
RUN mkdir -p /app/models/{gfpgan,gpen,codeformer,restoreformer,retinaface} && \
    mkdir -p /app/{input,output,temp}

# Create placeholder files for models (download models separately before building)
RUN echo "=== Creating Model Placeholders ===" && \
    # Create placeholder files - MODELS NEED TO BE PROVIDED SEPARATELY
    echo "# Model placeholder - download SCRFD RetinaFace model here" > /app/utils/scrfd_2.5g_bnkps.txt && \
    echo "# Model placeholder - download GFPGAN v1.4 ONNX model here" > /app/enhancers/GFPGAN/GFPGANv1.4.txt && \
    echo "# Model placeholder - download GPEN BFR-512 ONNX model here" > /app/enhancers/GPEN/GPEN-BFR-512.txt && \
    echo "# Model placeholder - download CodeFormer ONNX model here" > /app/enhancers/Codeformer/codeformer.txt && \
    echo "# Model placeholder - download RestoreFormer ONNX models here" > /app/enhancers/restoreformer/restoreformer.txt && \
    echo "✅ Model placeholders created"

# Note: Models must be provided separately due to licensing and size constraints
# Download the following models and place them in the respective directories:
# 1. RetinaFace SCRFD: utils/scrfd_2.5g_bnkps.onnx
# 2. GFPGAN v1.4: enhancers/GFPGAN/GFPGANv1.4.onnx  
# 3. GPEN BFR-512: enhancers/GPEN/GPEN-BFR-512.onnx
# 4. CodeFormer: enhancers/Codeformer/codeformer.onnx
# 5. RestoreFormer: enhancers/restoreformer/restoreformer16.onnx and restoreformer32.onnx

# Check model directories exist (models will be verified at runtime)
RUN echo "=== Checking Model Directories ===" && \
    test -d /app/utils && echo "✅ Utils directory OK" || echo "❌ Utils directory MISSING" && \
    test -d /app/enhancers/GFPGAN && echo "✅ GFPGAN directory OK" || echo "❌ GFPGAN directory MISSING" && \
    test -d /app/enhancers/GPEN && echo "✅ GPEN directory OK" || echo "❌ GPEN directory MISSING" && \
    test -d /app/enhancers/Codeformer && echo "✅ CodeFormer directory OK" || echo "❌ CodeFormer directory MISSING" && \
    test -d /app/enhancers/restoreformer && echo "✅ RestoreFormer directory OK" || echo "❌ RestoreFormer directory MISSING" && \
    echo "⚠️ IMPORTANT: Models must be added manually before deployment" && \
    echo "=== Directory Check Complete ==="

# Copy handler file
COPY face_enhancer_handler.py /app/face_enhancer_handler.py

# Make CLI executable
RUN chmod +x /app/face_enhancer_cli.py

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; import cv2; import onnxruntime; print('🚀 Ready')" || exit 1

# Expose port
EXPOSE 8000

# Run handler
CMD ["python", "face_enhancer_handler.py"]
