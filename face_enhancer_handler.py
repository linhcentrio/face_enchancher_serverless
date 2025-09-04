#!/usr/bin/env python3
"""
RunPod Serverless Handler cho Face Enhancement Service - GPU OPTIMIZED VERSION
Fixed: GPU detection, urllib3 compatibility, cURL issues, và performance
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import sys
import json
import traceback
import subprocess
import psutil
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "media.aiclip.ai")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "VtZ6MUPfyTOH3qSiohA2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# Initialize MinIO
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    minio_client.bucket_exists(MINIO_BUCKET)
    logger.info(f"✅ MinIO connected: {MINIO_ENDPOINT}/{MINIO_BUCKET}")
except Exception as e:
    logger.error(f"❌ MinIO failed: {e}")
    minio_client = None

# Enhanced enhancer info
SUPPORTED_ENHANCERS = {
    'gfpgan': {'name': 'GFPGAN v1.4', 'speed': 'fast', 'quality': 'good'},
    'gpen': {'name': 'GPEN-BFR-512', 'speed': 'medium', 'quality': 'high'}, 
    'codeformer': {'name': 'CodeFormer', 'speed': 'medium', 'quality': 'excellent'},
    'restoreformer16': {'name': 'RestoreFormer-16', 'speed': 'fast', 'quality': 'very-good'},
    'restoreformer32': {'name': 'RestoreFormer-32', 'speed': 'slow', 'quality': 'excellent'}
}

MODEL_PATHS = {
    'gfpgan': '/app/enhancers/GFPGAN/GFPGANv1.4.onnx',
    'gpen': '/app/enhancers/GPEN/GPEN-BFR-512.onnx',
    'codeformer': '/app/enhancers/Codeformer/codeformer.onnx',
    'restoreformer16': '/app/enhancers/restoreformer/restoreformer16.onnx',
    'restoreformer32': '/app/enhancers/restoreformer/restoreformer32.onnx',
    'retinaface': '/app/utils/scrfd_2.5g_bnkps.onnx'
}

def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information"""
    try:
        import torch
        
        gpu_info = {
            'torch_cuda_available': torch.cuda.is_available(),
            'torch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            gpu_info.update({
                'gpu_count': torch.cuda.device_count(),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            })
            
        # Check ONNX Runtime GPU providers
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            gpu_providers = [p for p in available_providers if any(gpu in p for gpu in ['CUDA', 'GPU', 'DML', 'TensorRT'])]
            
            gpu_info.update({
                'onnx_version': ort.__version__,
                'onnx_providers': available_providers,
                'onnx_gpu_providers': gpu_providers,
                'onnx_gpu_available': len(gpu_providers) > 0
            })
            
        except Exception as e:
            logger.warning(f"ONNX Runtime info error: {e}")
            
        return gpu_info
        
    except Exception as e:
        logger.error(f"GPU info error: {e}")
        return {'error': str(e)}

def create_optimized_session() -> requests.Session:
    """Create requests session with fixed urllib3 compatibility"""
    session = requests.Session()
    
    # Fixed retry strategy - compatible with urllib3 1.26.x
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Fixed: was method_whitelist
        backoff_factor=2,
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        'User-Agent': 'Face-Enhancement-Service/2.1',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    })
    
    return session

def download_file_robust(url: str, output_path: str, max_retries: int = 3) -> bool:
    """Enhanced download with multiple fallback methods"""
    
    # Method 1: Requests with fixed retry
    try:
        logger.info(f"📥 Downloading via requests: {url}")
        session = create_optimized_session()
        
        response = session.get(url, stream=True, timeout=(30, 300))
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
        if downloaded > 1024:  # At least 1KB
            logger.info(f"✅ Downloaded {downloaded/1024/1024:.1f}MB via requests")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ Requests download failed: {e}")

    # Method 2: wget fallback
    try:
        logger.info(f"📥 Downloading via wget: {url}")
        
        cmd = [
            'wget', '-q', '--tries=3', '--timeout=60',
            '--read-timeout=300', '-c', '--progress=bar:force',
            '-O', output_path, url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"✅ Downloaded {size_mb:.1f}MB via wget")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ wget download failed: {e}")

    # Method 3: curl fallback (fixed command)
    try:
        logger.info(f"📥 Downloading via curl: {url}")
        
        cmd = [
            'curl', '-L', '-C', '-', '--max-time', '600',
            '--connect-timeout', '30', '--retry', '3',
            '--silent', '--show-error', '--fail',
            '-o', output_path, url
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=700)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"✅ Downloaded {size_mb:.1f}MB via curl")
            return True
            
    except Exception as e:
        logger.warning(f"⚠️ curl download failed: {e}")

    return False

def verify_models() -> Tuple[bool, list, Dict[str, float]]:
    """Comprehensive model verification"""
    logger.info("🔍 Verifying models...")
    missing_models = []
    model_sizes = {}
    
    for name, path in MODEL_PATHS.items():
        try:
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                model_sizes[name] = round(size_mb, 1)
                logger.info(f"✅ {name}: {size_mb:.1f}MB")
            else:
                missing_models.append(f"{name}: {path}")
                logger.error(f"❌ Missing: {name}")
        except Exception as e:
            missing_models.append(f"{name}: {e}")
            logger.error(f"❌ Error checking {name}: {e}")
            
    success = len(missing_models) == 0
    total_size = sum(model_sizes.values())
    
    if success:
        logger.info(f"✅ All models verified: {total_size:.1f}MB total")
    else:
        logger.error(f"❌ {len(missing_models)} models missing")
        
    return success, missing_models, model_sizes

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload with error handling"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO not available")
            
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading {file_size:.1f}MB: {object_name}")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Uploaded: {url}")
        return url
        
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise

def get_file_type(file_path: str) -> str:
    """Fixed file type detection"""
    try:
        ext = os.path.splitext(file_path)[1].lower()  # Fixed: was [15]
        
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.webm']
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if ext in video_exts:
            return 'video'
        elif ext in image_exts:
            return 'image'
        else:
            return 'unknown'
    except:
        return 'unknown'

def run_face_enhancer(input_path: str, output_path: str, enhancer: str,
                     device: str = 'cuda', blend: float = 0.8,
                     codeformer_w: float = 0.9, skip_frames: int = 1) -> Tuple[bool, str, Dict[str, Any]]:
    """Run enhancement with GPU optimization"""
    try:
        logger.info(f"🎨 Enhancing with {enhancer} on {device}")
        
        # Auto-detect best device
        gpu_info = get_gpu_info()
        if device == 'cuda' and not gpu_info.get('onnx_gpu_available', False):
            logger.warning("⚠️ CUDA requested but not available, using CPU")
            device = 'cpu'
        elif device == 'cuda':
            logger.info(f"🚀 Using GPU: {gpu_info.get('gpu_name', 'Unknown')}")
            
        cmd = [
            'python', '/app/face_enhancer_cli.py',
            '--input', input_path,
            '--output', output_path, 
            '--enhancer', enhancer,
            '--device', device,
            '--blend', str(blend),
            '--server_mode', '--verbose'
        ]
        
        if enhancer == 'codeformer':
            cmd.extend(['--codeformer_w', str(codeformer_w)])
            
        if get_file_type(input_path) == 'video' and skip_frames > 1:
            cmd.extend(['--skip_frames', str(skip_frames)])
            
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        
        start_time = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd='/app'
        )
        
        # Real-time monitoring
        while process.poll() is None:
            # Timeout check (20 minutes)
            if time.time() - start_time > 1200:
                process.terminate()
                return False, "Timeout (20 min limit)", {}
                
            time.sleep(1)
            
        stdout, stderr = process.communicate()
        processing_time = time.time() - start_time
        
        stats = {
            'processing_time': round(processing_time, 2),
            'return_code': process.returncode,
            'device_used': device,
            'gpu_info': gpu_info
        }
        
        if process.returncode == 0 and os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            stats['output_size_mb'] = round(output_size, 2)
            
            logger.info(f"✅ Enhanced in {processing_time:.1f}s → {output_size:.1f}MB")
            return True, f"Success in {processing_time:.1f}s", stats
        else:
            error_msg = stderr.strip() if stderr else "Unknown error"
            logger.error(f"❌ Enhancement failed: {error_msg}")
            return False, error_msg, stats
            
    except Exception as e:
        logger.error(f"❌ Enhancement error: {e}")
        return False, str(e), {}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler with GPU optimization"""
    job_id = job.get("id", str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Job {job_id}: Starting enhancement")
        
        # Get GPU status
        gpu_info = get_gpu_info()
        logger.info(f"🖥️ GPU Status: {gpu_info.get('torch_cuda_available', False)} | ONNX GPU: {gpu_info.get('onnx_gpu_available', False)}")
        
        job_input = job.get("input", {})
        
        # Validate inputs
        if not job_input.get("input_url") or not job_input.get("enhancer"):
            return {"error": "Missing input_url or enhancer", "job_id": job_id}
            
        input_url = job_input["input_url"]
        enhancer = job_input["enhancer"]
        device = job_input.get("device", "cuda" if gpu_info.get('onnx_gpu_available') else "cpu")
        blend = float(job_input.get("blend", 0.8))
        skip_frames = int(job_input.get("skip_frames", 1))
        
        if enhancer not in SUPPORTED_ENHANCERS:
            return {"error": f"Unsupported enhancer: {enhancer}", "job_id": job_id}
            
        logger.info(f"🎨 Config: {enhancer} on {device}, blend={blend}")
        
        # Verify models
        models_ok, missing, model_sizes = verify_models()
        if not models_ok:
            return {"error": "Models missing", "missing_models": missing, "job_id": job_id}
            
        # Process file
        with tempfile.TemporaryDirectory(prefix=f"enhance_{job_id}_") as temp_dir:
            # Download
            parsed_url = urlparse(input_url)
            filename = os.path.basename(parsed_url.path) or f"input_{int(time.time())}.mp4"
            input_path = os.path.join(temp_dir, filename)
            
            logger.info("📥 Downloading input file...")
            download_start = time.time()
            
            if not download_file_robust(input_url, input_path):
                return {"error": "Download failed", "job_id": job_id}
                
            download_time = time.time() - download_start
            
            if not os.path.exists(input_path) or os.path.getsize(input_path) < 1024:
                return {"error": "Downloaded file invalid", "job_id": job_id}
                
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            file_type = get_file_type(input_path)
            
            logger.info(f"📊 Input: {file_type}, {input_size:.1f}MB, downloaded in {download_time:.1f}s")
            
            # Generate output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_{enhancer}_enhanced{'.mp4' if file_type == 'video' else ext}"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Enhance
            logger.info("🎨 Starting enhancement...")
            success, message, stats = run_face_enhancer(
                input_path, output_path, enhancer, device, blend, 0.9, skip_frames
            )
            
            if not success:
                return {"error": f"Enhancement failed: {message}", "job_id": job_id, "stats": stats}
                
            # Upload result
            logger.info("📤 Uploading result...")
            timestamp = int(time.time())
            object_name = f"enhanced/{enhancer}/{timestamp}/{job_id}_{output_filename}"
            
            try:
                output_url = upload_to_minio(output_path, object_name)
            except Exception as e:
                return {"error": f"Upload failed: {e}", "job_id": job_id}
                
            # Success response
            total_time = time.time() - start_time
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            
            logger.info(f"✅ Job {job_id} completed in {total_time:.1f}s")
            
            return {
                "output_url": output_url,
                "status": "completed",
                "job_id": job_id,
                "processing_time_seconds": round(total_time, 2),
                "file_info": {
                    "input_size_mb": round(input_size, 2),
                    "output_size_mb": round(output_size, 2),
                    "type": file_type
                },
                "config": {
                    "enhancer": enhancer,
                    "device_used": stats.get('device_used', device),
                    "blend": blend,
                    "skip_frames": skip_frames if file_type == 'video' else None
                },
                "performance": {
                    "download_time": round(download_time, 2),
                    "enhancement_time": stats.get('processing_time', 0),
                    "total_time": round(total_time, 2)
                },
                "gpu_info": gpu_info,
                "version": "2.1_GPU_OPTIMIZED"
            }
            
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Internal error: {str(e)}",
            "job_id": job_id,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

def health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Health check with GPU validation"""
    try:
        # Check GPU
        gpu_info = get_gpu_info()
        
        # Check models
        models_ok, missing, sizes = verify_models()
        
        # Check MinIO
        minio_ok = minio_client is not None
        
        health_info = {
            "gpu_info": gpu_info,
            "models": {"verified": models_ok, "total_size_mb": sum(sizes.values())},
            "storage": {"minio_available": minio_ok},
            "status": "healthy" if (models_ok and minio_ok) else "degraded"
        }
        
        if models_ok and minio_ok:
            logger.info("✅ Health check passed")
            return True, "All systems operational", health_info
        else:
            issues = []
            if not models_ok:
                issues.append(f"Missing {len(missing)} models")
            if not minio_ok:
                issues.append("MinIO unavailable")
            return False, "; ".join(issues), health_info
            
    except Exception as e:
        return False, f"Health check error: {e}", {"error": str(e)}

# Startup sequence
if __name__ == "__main__":
    logger.info("🚀 Face Enhancement Service - GPU Optimized v2.1")
    
    try:
        # Verify GPU setup
        gpu_info = get_gpu_info()
        logger.info(f"🔥 GPU Status:")
        logger.info(f"   PyTorch CUDA: {gpu_info.get('torch_cuda_available', False)}")
        logger.info(f"   ONNX GPU: {gpu_info.get('onnx_gpu_available', False)}")
        if gpu_info.get('gpu_name'):
            logger.info(f"   GPU: {gpu_info['gpu_name']}")
            
        # Health check
        health_ok, health_msg, health_info = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)
            
        logger.info(f"✅ {health_msg}")
        logger.info(f"🎨 Ready: {len(SUPPORTED_ENHANCERS)} enhancers, GPU: {gpu_info.get('onnx_gpu_available', False)}")
        
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)
