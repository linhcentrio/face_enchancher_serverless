# Face Enhancer Docker Image - RunPod Serverless with Direct HF Downloads
FROM spxiong/pytorch:2.1.1-py3.10.15-cuda12.1.0-ubuntu22.04

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg wget curl git unzip \
    build-essential python3-dev \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libgoogle-perftools4 libtcmalloc-minimal4 \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install PyTorch with CUDA 12.1 support
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.1.1 with CUDA 12.1 support
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 xformers==0.0.23 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --no-cache-dir

# Install core ML and processing libraries
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    opencv-python-headless>=4.5.0 \
    Pillow>=8.0.0 \
    scipy>=1.7.0 \
    scikit-image>=0.19.0 \
    onnxruntime-gpu==1.14.1 \
    tqdm>=4.60.0

# Install server and cloud libraries
RUN pip install --no-cache-dir \
    runpod>=1.5.0 \
    minio>=7.1.0 \
    requests>=2.25.0 \
    psutil

# Install InsightFace from specific wheel
RUN pip install --no-cache-dir \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/insightface-0.7.3-cp310-cp310-linux_x86_64.whl

# Copy application files FIRST (including utils with retinaface model)
COPY . /app/

# Create enhancer directories
RUN mkdir -p /app/enhancers/Codeformer \
    /app/enhancers/GFPGAN \
    /app/enhancers/GPEN \
    /app/enhancers/restoreformer

# Download models directly from HuggingFace URLs
RUN echo "=== Downloading Face Enhancement Models from HuggingFace ===" && \
    wget --progress=bar:force \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/Codeformer/codeformer.onnx \
    -O /app/enhancers/Codeformer/codeformer.onnx && \
    echo "✅ CodeFormer downloaded" && \
    \
    wget --progress=bar:force \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/GFPGAN/GFPGANv1.4.onnx \
    -O /app/enhancers/GFPGAN/GFPGANv1.4.onnx && \
    echo "✅ GFPGAN downloaded" && \
    \
    wget --progress=bar:force \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/GPEN/GPEN-BFR-512.onnx \
    -O /app/enhancers/GPEN/GPEN-BFR-512.onnx && \
    echo "✅ GPEN downloaded" && \
    \
    wget --progress=bar:force \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/restoreformer/restoreformer16.onnx \
    -O /app/enhancers/restoreformer/restoreformer16.onnx && \
    echo "✅ RestoreFormer16 downloaded" && \
    \
    wget --progress=bar:force \
    https://huggingface.co/manh-linh/face_enhancers_onnx/resolve/main/restoreformer/restoreformer32.onnx \
    -O /app/enhancers/restoreformer/restoreformer32.onnx && \
    echo "✅ RestoreFormer32 downloaded"

# Verify all downloaded models and existing retinaface
RUN echo "=== Verifying All Models ===" && \
    echo "📁 Utils directory:" && ls -lh /app/utils/ && \
    echo "📁 CodeFormer:" && ls -lh /app/enhancers/Codeformer/ && \
    echo "📁 GFPGAN:" && ls -lh /app/enhancers/GFPGAN/ && \
    echo "📁 GPEN:" && ls -lh /app/enhancers/GPEN/ && \
    echo "📁 RestoreFormer:" && ls -lh /app/enhancers/restoreformer/ && \
    echo "=== Model Verification Complete ==="

# Create additional required directories
RUN mkdir -p /app/{input,output,temp,logs}

# Set proper permissions
RUN chmod +x /app/face_enhancer_cli.py \
    && chmod +x /app/face_enhancer_handler.py

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')" \
    && python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" \
    && python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')" \
    && python -c "import insightface; print('InsightFace installed successfully')" \
    && echo "✅ All dependencies verified"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; import cv2; import onnxruntime; import insightface; print('🚀 Ready')" || exit 1

# Expose port
EXPOSE 8000

# Run handler
CMD ["python", "/app/face_enhancer_handler.py"]


