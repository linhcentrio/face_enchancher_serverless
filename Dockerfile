# Face Enhancer Docker Image - RunPod Serverless with GPU Support - FIXED
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# Environment variables for GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies with better GPU support
RUN apt-get update && apt-get install -y \
    ffmpeg wget curl git unzip \
    build-essential python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgoogle-perftools4 libtcmalloc-minimal4 \
    libgl1-mesa-glx libglib2.0-0 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install core packages
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.1 with CUDA 12.1 support (explicit)
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 xformers==0.0.23 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir --force-reinstall

# Install compatible versions for requests/urllib3
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    urllib3==1.26.18 \
    certifi>=2023.7.22

# Install core ML and processing libraries with GPU support
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python-headless>=4.5.0 \
    Pillow>=8.0.0 \
    scipy>=1.7.0 \
    scikit-image>=0.19.0 \
    tqdm>=4.60.0

# Install ONNX Runtime GPU with proper CUDA support
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.16.3 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install server and cloud libraries
RUN pip install --no-cache-dir \
    runpod>=1.5.0 \
    minio>=7.1.0 \
    psutil

# Install InsightFace from specific wheel
RUN pip install --no-cache-dir \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Copy application files FIRST
COPY . /app/

# Create enhancer directories
RUN mkdir -p /app/enhancers/Codeformer \
    /app/enhancers/GFPGAN \
    /app/enhancers/GPEN \
    /app/enhancers/restoreformer

# Download models with retry and compression
RUN echo "=== Downloading Face Enhancement Models ===" && \
    for i in 1 2 3; do \
        wget --timeout=30 --tries=3 --progress=bar:force \
        https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/Codeformer/codeformer.onnx \
        -O /app/enhancers/Codeformer/codeformer.onnx && break || sleep 10; \
    done && \
    echo "✅ CodeFormer downloaded" && \
    \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=3 --progress=bar:force \
        https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/GFPGAN/GFPGANv1.4.onnx \
        -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && break || sleep 10; \
    done && \
    echo "✅ GFPGAN downloaded" && \
    \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=3 --progress=bar:force \
        https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/GPEN/GPEN-BFR-512.onnx \
        -O /app/enhancers/GPEN/GPEN-BFR-512.onnx && break || sleep 10; \
    done && \
    echo "✅ GPEN downloaded" && \
    \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=3 --progress=bar:force \
        https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/restoreformer/restoreformer16.onnx \
        -O /app/enhancers/restoreformer/restoreformer16.onnx && break || sleep 10; \
    done && \
    echo "✅ RestoreFormer16 downloaded" && \
    \
    for i in 1 2 3; do \
        wget --timeout=60 --tries=3 --progress=bar:force \
        https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/restoreformer/restoreformer32.onnx \
        -O /app/enhancers/restoreformer/restoreformer32.onnx && break || sleep 10; \
    done && \
    echo "✅ RestoreFormer32 downloaded"

# Verify all models
RUN echo "=== Model Verification ===" && \
    echo "📁 Utils:" && ls -lh /app/utils/ && \
    echo "📁 CodeFormer:" && ls -lh /app/enhancers/Codeformer/ && \
    echo "📁 GFPGAN:" && ls -lh /app/enhancers/GFPGAN/ && \
    echo "📁 GPEN:" && ls -lh /app/enhancers/GPEN/ && \
    echo "📁 RestoreFormer:" && ls -lh /app/enhancers/restoreformer/

# Create directories and set permissions
RUN mkdir -p /app/{input,output,temp,logs} && \
    chmod +x /app/face_enhancer_cli.py && \
    chmod +x /app/face_enhancer_handler.py

# Comprehensive dependency verification with GPU check
RUN echo "=== Dependency Verification ===" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'No GPU')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}'); print(f'GPU Providers: {[p for p in ort.get_available_providers() if \"CUDA\" in p or \"GPU\" in p]}')" && \
    python -c "import insightface; print('InsightFace: OK')" && \
    python -c "import requests; print(f'Requests: {requests.__version__}')" && \
    echo "✅ All dependencies verified"

# Health check with GPU validation
HEALTHCHECK --interval=30s --timeout=20s --start-period=300s --retries=3 \
    CMD python -c "import torch; import onnxruntime; print('GPU:', torch.cuda.is_available()); print('ONNX GPU:', len([p for p in onnxruntime.get_available_providers() if 'CUDA' in p]) > 0)" || exit 1

EXPOSE 8000

CMD ["python", "/app/face_enhancer_handler.py"]



