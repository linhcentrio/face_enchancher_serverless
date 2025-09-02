# Face Enhancement Service - RunPod Serverless

🎬 **Dịch vụ nâng cao chất lượng khuôn mặt trong video/ảnh** được tối ưu cho triển khai serverless trên RunPod.

## ✨ Tính năng

- 🧠 **4 AI Models**: GFPGAN, GPEN, CodeFormer, RestoreFormer
- 🎬 **Video Processing**: Xử lý video frame-by-frame với face detection
- 🖼️ **Image Support**: Hỗ trợ xử lý ảnh đơn lẻ
- ⚡ **Performance Optimization**: Tối ưu cho CUDA/CPU
- 🎭 **Face Detection**: RetinaFace tự động detect và align khuôn mặt
- 📁 **Batch Processing**: Xử lý hàng loạt video/ảnh
- 🔊 **Audio Preservation**: Giữ nguyên audio từ video gốc
- 🌐 **Serverless Ready**: Tích hợp RunPod và MinIO storage

## 🚀 RunPod Deployment

### 1. Build Docker Image
```bash
docker build -t face-enhancer-serverless .
```

### 2. Deploy lên RunPod
- Upload Docker image lên Docker Hub/Registry
- Tạo RunPod Serverless Endpoint
- Configure với image của bạn

### 3. Test API
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://example.com/video.mp4",
      "enhancer": "gfpgan",
      "quality": "high"
    }
  }'
```

## 📁 Models Required

Tải các model từ [Google Drive](https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ) và đặt vào thư mục tương ứng:

```
enhancers/
├── GFPGAN/GFPGANv1.4.onnx
├── GPEN/GPEN-BFR-512.onnx  
├── Codeformer/codeformer.onnx
└── restoreformer/
    ├── restoreformer16.onnx
    └── restoreformer32.onnx

utils/
└── scrfd_2.5g_bnkps.onnx  # RetinaFace model
```

## 🛠️ Local Development

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Test locally
python face_enhancer_cli.py --input test.mp4 --output enhanced.mp4 --enhancer gfpgan

# Test handler locally
python face_enhancer_handler.py
```

## 📝 API Documentation

### Input Parameters:
- `video_url` (string): URL của video/ảnh cần enhance
- `enhancer` (string): loại enhancer [gfpgan, gpen, codeformer, restoreformer]
- `quality` (string): mức chất lượng [low, medium, high]
- `blend_factor` (float): mức độ blend (0.1-1.0)

### Output:
- `enhanced_url` (string): URL của video/ảnh đã được enhance
- `processing_time` (float): thời gian xử lý
- `status` (string): trạng thái xử lý

## 🔗 Credits

Face enhancers taken from: https://github.com/harisreedhar/Face-Upscalers-ONNX
Face detection taken from: https://github.com/neuralchen/SimSwap



