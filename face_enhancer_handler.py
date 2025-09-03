#!/usr/bin/env python3

"""
RunPod Serverless Handler cho Face Enhancement Service - Optimized
Tối ưu cho việc nâng cao chất lượng khuôn mặt trong video/ảnh
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
from pathlib import Path
from minio import Minio
from urllib.parse import quote, urlparse
import logging

# Configure logging for RunPod
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# MinIO Configuration
MINIO_ENDPOINT = "media.aiclip.ai"
MINIO_ACCESS_KEY = "VtZ6MUPfyTOH3qSiohA2"
MINIO_SECRET_KEY = "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t"
MINIO_BUCKET = "video"
MINIO_SECURE = False

# Initialize MinIO client with error handling
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    logger.info("✅ MinIO client initialized")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Supported enhancers
SUPPORTED_ENHANCERS = ['gfpgan', 'gpen', 'codeformer', 'restoreformer16', 'restoreformer32']

# Model paths for verification (updated for HuggingFace repo structure)
MODEL_PATHS = {
    'gfpgan': '/app/enhancers/GFPGAN/GFPGANv1.4.onnx',
    'gpen': '/app/enhancers/GPEN/GPEN-BFR-512.onnx',
    'codeformer': '/app/enhancers/Codeformer/codeformer.onnx',
    'restoreformer16': '/app/enhancers/restoreformer/restoreformer16.onnx',
    'restoreformer32': '/app/enhancers/restoreformer/restoreformer32.onnx',
    'retinaface': '/app/utils/scrfd_2.5g_bnkps.onnx'
}

def verify_models() -> tuple[bool, list]:
    """Verify all required models exist"""
    logger.info("🔍 Verifying face enhancement models...")
    missing_models = []
    existing_models = []
    total_size = 0

    # Check enhancers directory first
    if not os.path.exists('/app/enhancers'):
        logger.error("❌ Enhancers directory not found")
        return False, ["Enhancers directory missing"]

    # List contents of enhancers directory for debugging
    try:
        enhancer_contents = os.listdir('/app/enhancers')
        logger.info(f"📁 Enhancers directory contents: {enhancer_contents}")
    except Exception as e:
        logger.error(f"❌ Cannot list enhancers directory: {e}")

    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                file_size_mb = os.path.getsize(path) / (1024 * 1024)
                existing_models.append(f"{name}: {file_size_mb:.1f}MB")
                total_size += file_size_mb
                logger.info(f"✅ {name}: {file_size_mb:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Error checking {name}: {e}")
                missing_models.append(f"{name}: {path} (error reading)")
        else:
            missing_models.append(f"{name}: {path}")
            logger.error(f"❌ Missing: {name} at {path}")

    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)}/{len(MODEL_PATHS)} models")
        for model in missing_models:
            logger.error(f"   - {model}")
        return False, missing_models
    else:
        logger.info(f"✅ All {len(existing_models)} models verified! Total: {total_size:.1f}MB")
        return True, []

def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO storage"""
    try:
        if not minio_client:
            raise RuntimeError("MinIO client not initialized")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"📤 Uploading to MinIO: {object_name} ({file_size_mb:.1f}MB)")
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        
        file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
        logger.info(f"✅ Upload completed: {file_url}")
        return file_url

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise e

def download_input_file(url: str, output_path: str) -> bool:
    """Download input file from URL with better error handling"""
    try:
        logger.info(f"📥 Downloading input file from: {url}")
        
        # Add headers to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=300, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Downloaded: {file_size_mb:.1f}MB")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def get_file_type(file_path: str) -> str:
    """Determine if file is video or image"""
    ext = os.path.splitext(file_path)[1].lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if ext in video_extensions:
        return 'video'
    elif ext in image_extensions:
        return 'image'
    else:
        return 'unknown'

def run_face_enhancer(input_path: str, output_path: str, enhancer: str,
                     device: str = 'cuda', blend: float = 0.8,
                     codeformer_w: float = 0.9, skip_frames: int = 1) -> tuple[bool, str]:
    """Run face enhancer CLI via subprocess with improved error handling"""
    try:
        logger.info(f"🎨 Running face enhancer: {enhancer}")
        logger.info(f"📍 Input: {input_path}")
        logger.info(f"📍 Output: {output_path}")

        # Verify input file exists
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}"

        # Build command for face_enhancer_cli.py
        cmd = [
            'python', '/app/face_enhancer_cli.py',
            '--input', input_path,
            '--output', output_path,
            '--enhancer', enhancer,
            '--device', device,
            '--blend', str(blend),
            '--server_mode',
            '--verbose'  # Enable verbose for better debugging
        ]

        # Add enhancer-specific parameters
        if enhancer == 'codeformer':
            cmd.extend(['--codeformer_w', str(codeformer_w)])

        # Add skip frames for video (performance optimization)
        file_type = get_file_type(input_path)
        if file_type == 'video' and skip_frames > 1:
            cmd.extend(['--skip_frames', str(skip_frames)])

        logger.info(f"🔧 Command: {' '.join(cmd)}")

        # Run face enhancer with timeout
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
            check=False,
            cwd='/app'  # Ensure correct working directory
        )

        processing_time = time.time() - start_time

        # Log output for debugging
        if result.stdout:
            logger.info(f"📝 Enhancement stdout: {result.stdout}")
        if result.stderr:
            if result.returncode != 0:
                logger.error(f"❌ Enhancement stderr: {result.stderr}")
            else:
                logger.info(f"📝 Enhancement info: {result.stderr}")

        if result.returncode == 0:
            if os.path.exists(output_path):
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✅ Enhancement completed in {processing_time:.1f}s")
                logger.info(f"📊 Output size: {output_size_mb:.1f}MB")
                return True, f"Enhancement completed successfully in {processing_time:.1f}s"
            else:
                logger.error("❌ Enhancement completed but output file not found")
                return False, "Output file not created"
        else:
            logger.error(f"❌ Enhancement failed with return code: {result.returncode}")
            return False, f"Enhancement process failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        logger.error("❌ Enhancement process timed out")
        return False, "Processing timed out (30 minutes limit)"
    except Exception as e:
        logger.error(f"❌ Enhancement error: {e}")
        return False, f"Enhancement error: {str(e)}"

def validate_input_parameters(job_input: dict) -> tuple[bool, str]:
    """Validate input parameters with improved checks"""
    try:
        # Required parameters
        if "input_url" not in job_input or not job_input["input_url"]:
            return False, "Missing required parameter: input_url"
        
        if "enhancer" not in job_input or not job_input["enhancer"]:
            return False, "Missing required parameter: enhancer"

        # Validate enhancer
        enhancer = job_input["enhancer"]
        if enhancer not in SUPPORTED_ENHANCERS:
            return False, f"Unsupported enhancer. Must be one of: {', '.join(SUPPORTED_ENHANCERS)}"

        # Validate input URL with timeout
        input_url = job_input["input_url"]
        try:
            response = requests.head(input_url, timeout=10, allow_redirects=True)
            if response.status_code not in [200, 302]:
                return False, f"Input URL not accessible: {response.status_code}"
        except Exception as e:
            logger.warning(f"URL validation warning: {str(e)}")
            # Don't fail validation for URL check issues, proceed with download attempt

        # Validate numeric parameters
        blend = job_input.get("blend", 0.8)
        try:
            blend = float(blend)
            if not (0.0 <= blend <= 1.0):
                return False, "Blend factor must be between 0.0 and 1.0"
        except (ValueError, TypeError):
            return False, "Blend factor must be a valid number"

        codeformer_w = job_input.get("codeformer_w", 0.9)
        try:
            codeformer_w = float(codeformer_w)
            if not (0.0 <= codeformer_w <= 1.0):
                return False, "CodeFormer w parameter must be between 0.0 and 1.0"
        except (ValueError, TypeError):
            return False, "CodeFormer w parameter must be a valid number"

        skip_frames = job_input.get("skip_frames", 1)
        try:
            skip_frames = int(skip_frames)
            if not (1 <= skip_frames <= 5):
                return False, "Skip frames must be between 1 and 5"
        except (ValueError, TypeError):
            return False, "Skip frames must be a valid integer"

        return True, "Valid"

    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def handler(job):
    """
    Main RunPod handler for Face Enhancement Service - Optimized
    """
    job_id = job.get("id", str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Job {job_id}: Face Enhancement Started")
        
        job_input = job.get("input", {})
        logger.info(f"📋 Job input: {json.dumps(job_input, indent=2)}")

        # Validate input parameters
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            logger.error(f"❌ Validation failed: {validation_message}")
            return {
                "error": validation_message,
                "status": "failed",
                "job_id": job_id
            }

        # Extract parameters with type conversion
        input_url = job_input["input_url"]
        enhancer = job_input["enhancer"]
        device = job_input.get("device", "cuda")
        blend = float(job_input.get("blend", 0.8))
        codeformer_w = float(job_input.get("codeformer_w", 0.9))
        skip_frames = int(job_input.get("skip_frames", 1))

        logger.info(f"🎨 Enhancer: {enhancer}")
        logger.info(f"📱 Device: {device}")
        logger.info(f"🎯 Blend: {blend}")

        # Verify models before processing
        models_ok, missing_models = verify_models()
        if not models_ok:
            logger.error("❌ Models verification failed")
            return {
                "error": "Required models are missing",
                "missing_models": missing_models,
                "status": "failed",
                "job_id": job_id
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"📁 Using temp directory: {temp_dir}")
            
            # Determine file extension from URL
            parsed_url = urlparse(input_url)
            input_filename = os.path.basename(parsed_url.path)
            if not input_filename or '.' not in input_filename:
                input_filename = "input_file.mp4"
            
            input_path = os.path.join(temp_dir, input_filename)

            # Generate output filename
            name, ext = os.path.splitext(input_filename)
            if get_file_type(input_filename) == 'video':
                output_filename = f"{name}_enhanced.mp4"
            else:
                output_filename = f"{name}_enhanced{ext}"
            
            output_path = os.path.join(temp_dir, output_filename)

            # Download input file
            logger.info("📥 Downloading input file...")
            if not download_input_file(input_url, input_path):
                return {
                    "error": "Failed to download input file",
                    "status": "failed",
                    "job_id": job_id
                }

            # Get file info
            file_type = get_file_type(input_path)
            input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            logger.info(f"📊 Input file: {file_type}, {input_size_mb:.1f}MB")

            # Run face enhancement
            logger.info("🎨 Running face enhancement...")
            enhancement_start = time.time()
            
            success, message = run_face_enhancer(
                input_path=input_path,
                output_path=output_path,
                enhancer=enhancer,
                device=device,
                blend=blend,
                codeformer_w=codeformer_w,
                skip_frames=skip_frames
            )

            enhancement_time = time.time() - enhancement_start

            if not success:
                logger.error(f"❌ Enhancement failed: {message}")
                return {
                    "error": f"Face enhancement failed: {message}",
                    "status": "failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "job_id": job_id
                }

            # Upload result to MinIO
            logger.info("📤 Uploading result to storage...")
            output_object_name = f"face_enhanced_{job_id}_{uuid.uuid4().hex[:8]}_{output_filename}"
            
            try:
                output_url = upload_to_minio(output_path, output_object_name)
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                return {
                    "error": f"Failed to upload result: {str(e)}",
                    "status": "failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "job_id": job_id
                }

            # Calculate final statistics
            total_time = time.time() - start_time
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Total time: {total_time:.1f}s (enhancement: {enhancement_time:.1f}s)")
            logger.info(f"📊 Output: {output_size_mb:.1f}MB")

            return {
                "output_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "enhancement_time_seconds": round(enhancement_time, 2),
                "file_info": {
                    "type": file_type,
                    "input_size_mb": round(input_size_mb, 2),
                    "output_size_mb": round(output_size_mb, 2),
                    "enhancement_ratio": round(output_size_mb / input_size_mb, 2) if input_size_mb > 0 else 1.0
                },
                "enhancement_params": {
                    "enhancer": enhancer,
                    "device": device,
                    "blend_factor": blend,
                    "codeformer_w": codeformer_w if enhancer == 'codeformer' else None,
                    "skip_frames": skip_frames if file_type == 'video' else None
                },
                "status": "completed",
                "job_id": job_id
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "error": error_msg,
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }

def health_check():
    """Health check function with comprehensive validation"""
    try:
        # Check basic imports
        import torch
        import cv2
        import onnxruntime
        import insightface

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA not available, using CPU")

        # Check models
        models_ok, missing = verify_models()
        if not models_ok:
            return False, f"Missing models: {len(missing)}"

        # Check MinIO
        if not minio_client:
            return False, "MinIO not available"

        # Check CLI tool
        if not os.path.exists('/app/face_enhancer_cli.py'):
            return False, "Face enhancer CLI not found"

        return True, "All systems operational"

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting Face Enhancement Serverless Worker...")
    
    try:
        # Import and verify dependencies
        import torch
        import cv2
        import onnxruntime
        import insightface

        logger.info(f"🔥 PyTorch: {torch.__version__}")
        logger.info(f"🎯 CUDA Available: {torch.cuda.is_available()}")
        logger.info(f"📷 OpenCV: {cv2.__version__}")
        logger.info(f"🧠 ONNX Runtime: {onnxruntime.__version__}")
        logger.info(f"👤 InsightFace: {insightface.__version__}")

        if torch.cuda.is_available():
            logger.info(f"💾 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Health check on startup
        health_ok, health_msg = health_check()
        if not health_ok:
            logger.error(f"❌ Health check failed: {health_msg}")
            sys.exit(1)

        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎨 Ready to process face enhancement requests...")
        logger.info(f"🔧 Supported enhancers: {', '.join(SUPPORTED_ENHANCERS)}")

        # Start RunPod worker
        runpod.serverless.start({"handler": handler})

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

