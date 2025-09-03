#!/usr/bin/env python3
"""
RunPod Serverless Handler cho Face Enhancement Service - Production Optimized v2.0
Tối ưu với enhanced download retry logic và network resilience
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure comprehensive logging for RunPod with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/handler.log', mode='a') if os.path.exists('/app/logs') else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# MinIO Configuration with environment variable support
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "media.aiclip.ai")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "VtZ6MUPfyTOH3qSiohA2")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

# Initialize MinIO client with comprehensive error handling
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    # Test connection
    minio_client.bucket_exists(MINIO_BUCKET)
    logger.info(f"✅ MinIO client initialized successfully - {MINIO_ENDPOINT}/{MINIO_BUCKET}")
except Exception as e:
    logger.error(f"❌ MinIO initialization failed: {e}")
    minio_client = None

# Supported enhancers with detailed metadata
SUPPORTED_ENHANCERS = {
    'gfpgan': {
        'name': 'GFPGAN v1.4',
        'speed': 'fast',
        'quality': 'good',
        'description': 'General-purpose face enhancement with balanced speed/quality'
    },
    'gpen': {
        'name': 'GPEN-BFR-512',
        'speed': 'medium',
        'quality': 'high',
        'description': 'High detail enhancement, good for professional content'
    },
    'codeformer': {
        'name': 'CodeFormer',
        'speed': 'medium',
        'quality': 'excellent',
        'description': 'Smart restoration with identity preservation control'
    },
    'restoreformer16': {
        'name': 'RestoreFormer-16',
        'speed': 'fast',
        'quality': 'very-good',
        'description': 'Fast processing with very good quality'
    },
    'restoreformer32': {
        'name': 'RestoreFormer-32',
        'speed': 'slow',
        'quality': 'excellent',
        'description': 'Best quality but slower processing'
    }
}

# Model paths with FIXED retinaface path
MODEL_PATHS = {
    'gfpgan': '/app/enhancers/GFPGAN/GFPGANv1.4.onnx',
    'gpen': '/app/enhancers/GPEN/GPEN-BFR-512.onnx',
    'codeformer': '/app/enhancers/Codeformer/codeformer.onnx',
    'restoreformer16': '/app/enhancers/restoreformer/restoreformer16.onnx',
    'restoreformer32': '/app/enhancers/restoreformer/restoreformer32.onnx',
    'retinaface': '/app/utils/scrfd_2.5g_bnkps.onnx'  # ✅ FIXED PATH
}

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for monitoring"""
    try:
        import torch
        
        system_info = {
            'timestamp': time.time(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'memory_percent_used': psutil.virtual_memory().percent,
            'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2),
            'pytorch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            system_info.update({
                'cuda_available': True,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                'gpu_memory_allocated_gb': round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                'gpu_memory_cached_gb': round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            })
        else:
            system_info['cuda_available'] = False
            
        return system_info
        
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
        return {'error': str(e)}

def verify_models() -> Tuple[bool, list, Dict[str, float]]:
    """
    Comprehensive model verification with detailed reporting and integrity checks
    Returns: (success, missing_models, model_sizes)
    """
    logger.info("🔍 Starting comprehensive model verification...")
    missing_models = []
    existing_models = {}
    model_sizes = {}
    total_size = 0

    # Check required directories first
    required_dirs = ['/app/enhancers', '/app/utils']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            error_msg = f"Critical directory missing: {dir_path}"
            logger.error(f"❌ {error_msg}")
            return False, [error_msg], {}

    # Log directory contents for debugging
    try:
        enhancer_contents = os.listdir('/app/enhancers')
        utils_contents = os.listdir('/app/utils')
        logger.info(f"📁 Enhancers directory: {enhancer_contents}")
        logger.info(f"📁 Utils directory: {utils_contents}")
    except Exception as e:
        logger.warning(f"Could not list directories: {e}")

    # Verify each model with comprehensive checks
    for name, path in MODEL_PATHS.items():
        try:
            if os.path.exists(path):
                # Check file size and accessibility
                file_size_bytes = os.path.getsize(path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                
                # Validate file integrity
                if file_size_bytes == 0:
                    missing_models.append(f"{name}: {path} (empty file)")
                    logger.error(f"❌ Empty file: {name} at {path}")
                elif file_size_mb < 0.5:  # Models should be at least 0.5MB
                    missing_models.append(f"{name}: {path} (file too small: {file_size_mb:.1f}MB)")
                    logger.error(f"❌ File too small: {name} at {path} ({file_size_mb:.1f}MB)")
                elif not os.access(path, os.R_OK):
                    missing_models.append(f"{name}: {path} (not readable)")
                    logger.error(f"❌ File not readable: {name} at {path}")
                else:
                    # Verify file can be opened (basic integrity check)
                    try:
                        with open(path, 'rb') as f:
                            # Read first few bytes to verify file integrity
                            header = f.read(16)
                            if len(header) < 16:
                                missing_models.append(f"{name}: {path} (corrupted header)")
                                logger.error(f"❌ Corrupted file header: {name}")
                                continue
                        
                        existing_models[name] = file_size_mb
                        model_sizes[name] = file_size_mb
                        total_size += file_size_mb
                        logger.info(f"✅ {name}: {file_size_mb:.1f}MB")
                        
                    except Exception as e:
                        missing_models.append(f"{name}: {path} (read error: {str(e)})")
                        logger.error(f"❌ Cannot read file: {name} - {e}")
            else:
                missing_models.append(f"{name}: {path}")
                logger.error(f"❌ Missing: {name} at {path}")
                
                # Debug: Check parent directory
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    try:
                        files = os.listdir(parent_dir)
                        similar_files = [f for f in files if f.lower().endswith('.onnx')]
                        logger.info(f"📁 {parent_dir} ONNX files: {similar_files}")
                    except Exception as e:
                        logger.warning(f"Could not list {parent_dir}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Error checking {name}: {e}")
            missing_models.append(f"{name}: {path} (verification error: {str(e)})")

    # Summary logging with detailed statistics
    if missing_models:
        logger.error(f"❌ Missing {len(missing_models)}/{len(MODEL_PATHS)} models")
        for model in missing_models:
            logger.error(f"   - {model}")
        return False, missing_models, model_sizes
    else:
        logger.info(f"✅ All {len(existing_models)} models verified successfully!")
        logger.info(f"💾 Total model storage: {total_size:.1f}MB")
        logger.info(f"📊 Model breakdown: {', '.join([f'{k}={v:.1f}MB' for k,v in model_sizes.items()])}")
        return True, [], model_sizes

def create_robust_session() -> requests.Session:
    """Create a robust requests session with retry strategy"""
    session = requests.Session()
    
    # Enhanced retry strategy
    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=2,
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
        socket_options=[(6, 1, 1)]  # TCP_NODELAY
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Enhanced headers
    session.headers.update({
        'User-Agent': 'Face-Enhancement-Service/2.0 (Linux; x64; AI Processing)',
        'Accept': '*/*',
        'Accept-Encoding': 'identity',  # Disable compression to avoid issues
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    })
    
    return session

def download_input_file(url: str, output_path: str, max_retries: int = 5, timeout: int = 300) -> bool:
    """
    Enhanced download with retry logic, resume capability and multiple fallback methods
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Download attempt {attempt + 1}/{max_retries}: {url}")
            
            # Check if partial file exists for resume
            resume_pos = 0
            if os.path.exists(output_path) and attempt > 0:
                resume_pos = os.path.getsize(output_path)
                if resume_pos > 0:
                    logger.info(f"🔄 Resuming download from byte {resume_pos:,}")
            
            # Create robust session
            session = create_robust_session()
            
            # Set range header for resume
            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'
            
            start_time = time.time()
            
            # Make request with progressive timeout
            connect_timeout = min(30, 10 + attempt * 5)  # Increase timeout per attempt
            read_timeout = min(timeout, 60 + attempt * 30)
            
            response = session.get(
                url,
                headers=headers,
                timeout=(connect_timeout, read_timeout),
                stream=True,
                allow_redirects=True
            )
            
            # Check response status
            if response.status_code == 416:  # Range Not Satisfiable
                logger.warning("🔄 Range not satisfiable, starting fresh download")
                if os.path.exists(output_path):
                    os.remove(output_path)
                resume_pos = 0
                continue
                
            response.raise_for_status()
            
            # Get content info
            content_length = response.headers.get('content-length')
            total_size = int(content_length) if content_length else 0
            if resume_pos > 0:
                total_size += resume_pos
                
            logger.info(f"📊 File info: {total_size:,} bytes total, starting from {resume_pos:,}")
            
            downloaded_size = resume_pos
            last_log_time = time.time()
            
            # Write mode: append if resuming, write if new
            mode = 'ab' if resume_pos > 0 else 'wb'
            
            with open(output_path, mode) as f:
                chunk_size = 32768  # 32KB chunks for better performance
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress logging every 5 seconds or 1MB
                        current_time = time.time()
                        if (current_time - last_log_time > 5) or (downloaded_size % (1024*1024) == 0):
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                speed = (downloaded_size - resume_pos) / (current_time - start_time) / 1024  # KB/s
                                logger.info(f"📊 Progress: {progress:.1f}% ({downloaded_size:,}/{total_size:,} bytes, {speed:.1f} KB/s)")
                            last_log_time = current_time

            download_time = time.time() - start_time
            final_size = os.path.getsize(output_path)
            file_size_mb = final_size / (1024 * 1024)
            
            # Verify download completeness
            if total_size > 0 and final_size < total_size:
                logger.warning(f"⚠️ File incomplete: {final_size:,}/{total_size:,} bytes")
                if attempt < max_retries - 1:
                    wait_time = min(10, 2 ** attempt)  # Exponential backoff with max 10s
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("❌ File incomplete after all retries")
                    return False
            
            # Verify file is not empty and reasonable
            if final_size < 1024:  # Less than 1KB is suspicious
                logger.error(f"❌ Downloaded file too small: {final_size} bytes")
                if attempt < max_retries - 1:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    time.sleep(2 ** attempt)
                    continue
                return False
            
            download_speed = (final_size / (1024 * 1024)) / download_time if download_time > 0 else 0
            logger.info(f"✅ Download completed: {file_size_mb:.1f}MB in {download_time:.1f}s ({download_speed:.1f}MB/s)")
            return True
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout) as e:
            logger.warning(f"⚠️ Network error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            
            if attempt < max_retries - 1:
                wait_time = min(15, 2 ** attempt)  # Exponential backoff with max 15s
                logger.info(f"🔄 Network retry in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ All {max_retries} download attempts failed due to network issues")
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP error: {e} (Status: {e.response.status_code if e.response else 'Unknown'})")
            if e.response and e.response.status_code in [404, 403, 401]:
                # Don't retry for client errors
                return False
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"❌ Unexpected download error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"❌ Download failed after {max_retries} attempts")
                
    return False

def download_with_curl(url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    Fallback download method using curl with comprehensive retry logic
    """
    try:
        logger.info(f"🔄 Attempting cURL download: {url}")
        
        # Remove partial file for fresh start
        if os.path.exists(output_path):
            os.remove(output_path)
        
        for attempt in range(max_retries):
            cmd = [
                'curl',
                '-L',                    # Follow redirects
                '-C', '-',              # Resume capability
                '--max-time', '600',    # 10 minute total timeout
                '--connect-timeout', '30', # 30s connection timeout
                '--retry', '3',         # Built-in retries
                '--retry-delay', '2',   # 2s delay between retries
                '--retry-max-time', '300', # Max retry time
                '--fail',               # Fail silently on HTTP errors
                '--progress-bar',       # Show progress
                '--buffer-size', '32768', # 32KB buffer
                '-o', output_path,
                url
            ]
            
            logger.info(f"🔧 cURL attempt {attempt + 1}/{max_retries}: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=900  # 15 minute total timeout
            )
            
            download_time = time.time() - start_time
            
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1024:  # At least 1KB
                    file_size_mb = file_size / (1024 * 1024)
                    speed = file_size_mb / download_time if download_time > 0 else 0
                    logger.info(f"✅ cURL download successful: {file_size_mb:.1f}MB in {download_time:.1f}s ({speed:.1f}MB/s)")
                    return True
                else:
                    logger.warning(f"⚠️ cURL downloaded file too small: {file_size} bytes")
            
            # Log curl error details
            if result.stderr:
                logger.warning(f"⚠️ cURL stderr: {result.stderr.strip()}")
            if result.stdout:
                logger.info(f"📝 cURL stdout: {result.stdout.strip()}")
                
            if attempt < max_retries - 1:
                wait_time = 5 + attempt * 5
                logger.info(f"🔄 cURL retry in {wait_time}s...")
                time.sleep(wait_time)
                if os.path.exists(output_path):
                    os.remove(output_path)
            
        logger.error("❌ All cURL attempts failed")
        return False
        
    except subprocess.TimeoutExpired:
        logger.error("❌ cURL download timed out")
        return False
    except Exception as e:
        logger.error(f"❌ cURL download error: {e}")
        return False

def download_with_wget(url: str, output_path: str) -> bool:
    """
    Secondary fallback using wget
    """
    try:
        logger.info(f"🔄 Attempting wget download: {url}")
        
        cmd = [
            'wget',
            '--no-verbose',
            '--tries=3',
            '--timeout=30',
            '--read-timeout=300',
            '--continue',           # Resume capability
            '--progress=bar',
            '-O', output_path,
            url
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        download_time = time.time() - start_time
        
        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1024:
                file_size_mb = file_size / (1024 * 1024)
                logger.info(f"✅ wget download successful: {file_size_mb:.1f}MB in {download_time:.1f}s")
                return True
        
        if result.stderr:
            logger.warning(f"⚠️ wget error: {result.stderr}")
            
        return False
        
    except Exception as e:
        logger.error(f"❌ wget download error: {e}")
        return False

def upload_to_minio(local_path: str, object_name: str, max_retries: int = 3) -> str:
    """Upload file to MinIO storage with retry logic and progress tracking"""
    
    for attempt in range(max_retries):
        try:
            if not minio_client:
                raise RuntimeError("MinIO client not initialized")
                
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"📤 Upload attempt {attempt + 1}/{max_retries}: {object_name} ({file_size_mb:.1f}MB)")
            
            start_time = time.time()
            minio_client.fput_object(
                MINIO_BUCKET, 
                object_name, 
                local_path,
                metadata={'uploaded-by': 'face-enhancement-service'}
            )
            upload_time = time.time() - start_time
            upload_speed = file_size_mb / upload_time if upload_time > 0 else 0
            
            # Verify upload
            try:
                obj_stat = minio_client.stat_object(MINIO_BUCKET, object_name)
                uploaded_size_mb = obj_stat.size / (1024 * 1024)
                
                if abs(uploaded_size_mb - file_size_mb) > 0.1:  # Size mismatch > 100KB
                    logger.warning(f"⚠️ Size mismatch: local={file_size_mb:.1f}MB, uploaded={uploaded_size_mb:.1f}MB")
                    if attempt < max_retries - 1:
                        continue
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not verify upload: {e}")
            
            file_url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
            logger.info(f"✅ Upload completed in {upload_time:.1f}s ({upload_speed:.1f}MB/s): {file_url}")
            return file_url
            
        except Exception as e:
            logger.error(f"❌ Upload attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"🔄 Upload retry in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error("❌ All upload attempts failed")
                raise e

def get_file_type(file_path: str) -> str:
    """Determine if file is video, image or unknown with enhanced detection"""
    ext = os.path.splitext(file_path)[1].lower()
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif', '.tga']
    
    if ext in video_extensions:
        return 'video'
    elif ext in image_extensions:
        return 'image'
    else:
        # Try to detect by content if extension is unknown
        if os.path.exists(file_path):
            try:
                import magic
                mime_type = magic.from_file(file_path, mime=True)
                if mime_type.startswith('video/'):
                    return 'video'
                elif mime_type.startswith('image/'):
                    return 'image'
            except:
                pass
        return 'unknown'

def run_face_enhancer(input_path: str, output_path: str, enhancer: str,
                     device: str = 'cuda', blend: float = 0.8,
                     codeformer_w: float = 0.9, skip_frames: int = 1) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Run face enhancer CLI via subprocess with comprehensive monitoring and error handling
    Returns: (success, message, stats)
    """
    try:
        logger.info(f"🎨 Starting face enhancement with {enhancer}")
        logger.info(f"📍 Input: {input_path} ({os.path.getsize(input_path) / (1024*1024):.1f}MB)")
        logger.info(f"📍 Output: {output_path}")
        
        # Pre-flight checks
        if not os.path.exists(input_path):
            return False, f"Input file not found: {input_path}", {}
            
        if not os.access(input_path, os.R_OK):
            return False, f"Input file not readable: {input_path}", {}
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build comprehensive command
        cmd = [
            'python', '/app/face_enhancer_cli.py',
            '--input', input_path,
            '--output', output_path,
            '--enhancer', enhancer,
            '--device', device,
            '--blend', str(blend),
            '--server_mode',
            '--verbose'
        ]

        # Add enhancer-specific parameters
        if enhancer == 'codeformer':
            cmd.extend(['--codeformer_w', str(codeformer_w)])

        # Add skip frames for video performance optimization
        file_type = get_file_type(input_path)
        if file_type == 'video' and skip_frames > 1:
            cmd.extend(['--skip_frames', str(skip_frames)])
            logger.info(f"🚀 Video mode with skip_frames={skip_frames} for faster processing")

        logger.info(f"🔧 Enhancement command: {' '.join(cmd)}")

        # Monitor system resources before processing
        process_start_time = time.time()
        initial_memory = psutil.virtual_memory()
        initial_cpu = psutil.cpu_percent(interval=1)

        # Run enhancement process with comprehensive monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/app',
            env=dict(os.environ, PYTHONUNBUFFERED='1')
        )

        # Real-time output monitoring
        stdout_lines = []
        stderr_lines = []
        
        while True:
            # Check if process is still running
            poll = process.poll()
            
            # Read available output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    stdout_lines.append(line)
                    logger.info(f"📝 CLI: {line}")
            
            if process.stderr:
                line = process.stderr.readline()
                if line:
                    line = line.strip()
                    stderr_lines.append(line)
                    if "error" in line.lower() or "failed" in line.lower():
                        logger.error(f"❌ CLI Error: {line}")
                    else:
                        logger.info(f"📝 CLI Info: {line}")
            
            # Break if process finished and no more output
            if poll is not None and not line:
                break
                
            # Timeout check (30 minutes)
            if time.time() - process_start_time > 1800:
                logger.error("❌ Enhancement process timeout (30 minutes)")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                return False, "Processing timed out (30 minutes limit)", {}

        processing_time = time.time() - process_start_time
        final_memory = psutil.virtual_memory()
        final_cpu = psutil.cpu_percent()

        # Collect final output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            for line in remaining_stdout.strip().split('\n'):
                if line.strip():
                    stdout_lines.append(line.strip())
                    logger.info(f"📝 CLI Final: {line.strip()}")
                    
        if remaining_stderr:
            for line in remaining_stderr.strip().split('\n'):
                if line.strip():
                    stderr_lines.append(line.strip())
                    logger.info(f"📝 CLI Error Final: {line.strip()}")

        # Process completion analysis
        stats = {
            'processing_time': round(processing_time, 2),
            'memory_usage_change_mb': round((final_memory.used - initial_memory.used) / (1024*1024), 2),
            'cpu_usage_change': round(final_cpu - initial_cpu, 2),
            'return_code': process.returncode,
            'stdout_lines': len(stdout_lines),
            'stderr_lines': len(stderr_lines)
        }

        if process.returncode == 0:
            if os.path.exists(output_path):
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
                
                stats.update({
                    'output_size_mb': round(output_size_mb, 2),
                    'size_ratio': round(output_size_mb / input_size_mb, 2) if input_size_mb > 0 else 1.0,
                    'processing_speed_mbps': round(input_size_mb / processing_time, 2) if processing_time > 0 else 0
                })
                
                logger.info(f"✅ Enhancement completed successfully in {processing_time:.1f}s")
                logger.info(f"📊 Output: {output_size_mb:.1f}MB (ratio: {stats['size_ratio']:.2f}x)")
                return True, f"Enhancement completed in {processing_time:.1f}s", stats
            else:
                logger.error("❌ Enhancement completed but output file not found")
                return False, "Output file not created despite successful completion", stats
        else:
            error_details = {
                'return_code': process.returncode,
                'stderr_sample': stderr_lines[-10:] if stderr_lines else [],
                'stdout_sample': stdout_lines[-5:] if stdout_lines else []
            }
            
            error_msg = f"Enhancement failed (code {process.returncode})"
            if stderr_lines:
                error_msg += f": {stderr_lines[-1]}"
                
            logger.error(f"❌ {error_msg}")
            logger.error(f"🔍 Error details: {json.dumps(error_details, indent=2)}")
            
            stats['error_details'] = error_details
            return False, error_msg, stats

    except Exception as e:
        logger.error(f"❌ Enhancement error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return False, f"Enhancement error: {str(e)}", {'exception': str(e)}

def validate_input_parameters(job_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Comprehensive input parameter validation with detailed feedback"""
    try:
        # Required parameters validation
        required_params = ['input_url', 'enhancer']
        for param in required_params:
            if param not in job_input or not job_input[param]:
                return False, f"Missing required parameter: {param}"

        # Validate enhancer
        enhancer = job_input["enhancer"]
        if enhancer not in SUPPORTED_ENHANCERS:
            available = ', '.join(SUPPORTED_ENHANCERS.keys())
            return False, f"Unsupported enhancer '{enhancer}'. Available: {available}"

        # Validate input URL format
        input_url = job_input["input_url"]
        parsed_url = urlparse(input_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return False, f"Invalid URL format: {input_url}"

        # Basic URL accessibility check (non-blocking)
        try:
            response = requests.head(input_url, timeout=15, allow_redirects=True)
            if response.status_code >= 400:
                logger.warning(f"⚠️ URL returned status {response.status_code}, but continuing...")
        except Exception as e:
            logger.warning(f"⚠️ URL validation warning (continuing): {str(e)}")

        # Validate and convert numeric parameters
        numeric_validations = [
            ('blend', 0.0, 1.0, 0.8, "Blend factor must be between 0.0 and 1.0"),
            ('codeformer_w', 0.0, 1.0, 0.9, "CodeFormer w parameter must be between 0.0 and 1.0"),
        ]
        
        for param_name, min_val, max_val, default_val, error_msg in numeric_validations:
            if param_name in job_input:
                try:
                    value = float(job_input[param_name])
                    if not (min_val <= value <= max_val):
                        return False, f"{error_msg} (received: {value})"
                except (ValueError, TypeError):
                    return False, f"{param_name} must be a valid number (received: {job_input[param_name]})"

        # Validate skip_frames
        if 'skip_frames' in job_input:
            try:
                skip_frames = int(job_input['skip_frames'])
                if not (1 <= skip_frames <= 10):
                    return False, f"Skip frames must be between 1 and 10 (received: {skip_frames})"
            except (ValueError, TypeError):
                return False, f"Skip frames must be a valid integer (received: {job_input['skip_frames']})"

        # Validate device
        device = job_input.get('device', 'cuda')
        if device not in ['cpu', 'cuda']:
            return False, f"Device must be 'cpu' or 'cuda' (received: {device})"

        # Validate file extension if detectable from URL
        try:
            filename = os.path.basename(parsed_url.path)
            if filename:
                file_type = get_file_type(filename)
                if file_type == 'unknown':
                    logger.warning(f"⚠️ Unknown file type for: {filename}")
        except:
            pass

        logger.info("✅ All input parameters validated successfully")
        return True, "Valid"
        
    except Exception as e:
        logger.error(f"❌ Parameter validation error: {e}")
        return False, f"Parameter validation error: {str(e)}"

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler for Face Enhancement Service - Production Ready v2.0
    Enhanced with robust download retry logic and comprehensive error handling
    """
    job_id = job.get("id", str(uuid.uuid4()))
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Job {job_id}: Face Enhancement Started")
        
        # Log system status
        system_info = get_system_info()
        logger.info(f"🖥️ System Status: {system_info.get('memory_available_gb', 0):.1f}GB RAM available, {system_info.get('disk_free_gb', 0):.1f}GB disk free")
        
        job_input = job.get("input", {})
        logger.info(f"📋 Job input: {json.dumps(job_input, indent=2)}")

        # Comprehensive input validation
        is_valid, validation_message = validate_input_parameters(job_input)
        if not is_valid:
            logger.error(f"❌ Validation failed: {validation_message}")
            return {
                "error": validation_message,
                "status": "validation_failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "system_info": system_info
            }

        # Extract and convert parameters with defaults
        input_url = job_input["input_url"]
        enhancer = job_input["enhancer"]
        device = job_input.get("device", "cuda")
        blend = float(job_input.get("blend", 0.8))
        codeformer_w = float(job_input.get("codeformer_w", 0.9))
        skip_frames = int(job_input.get("skip_frames", 1))

        logger.info(f"🎨 Enhancement config: {enhancer} ({SUPPORTED_ENHANCERS[enhancer]['name']}) on {device}")
        logger.info(f"🎯 Parameters: blend={blend}, skip_frames={skip_frames}")
        if enhancer == 'codeformer':
            logger.info(f"🎯 CodeFormer w={codeformer_w}")

        # Pre-flight model verification
        models_ok, missing_models, model_sizes = verify_models()
        if not models_ok:
            logger.error("❌ Model verification failed")
            return {
                "error": "Required models are missing or corrupted",
                "missing_models": missing_models,
                "status": "model_verification_failed",
                "job_id": job_id,
                "processing_time_seconds": round(time.time() - start_time, 2),
                "system_info": system_info
            }

        logger.info(f"✅ Models verified: {len(model_sizes)} models, {sum(model_sizes.values()):.1f}MB total")

        # File processing in temporary directory with job-specific naming
        temp_prefix = f"face_enhance_{job_id}_{int(time.time())}_"
        with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
            logger.info(f"📁 Working directory: {temp_dir}")

            # Smart filename handling with sanitization
            parsed_url = urlparse(input_url)
            input_filename = os.path.basename(parsed_url.path)
            
            # Sanitize filename and add fallback
            if not input_filename or '.' not in input_filename:
                timestamp = int(time.time())
                input_filename = f"input_{timestamp}.mp4"
            
            # Remove any problematic characters
            input_filename = "".join(c for c in input_filename if c.isalnum() or c in '._-')
            input_path = os.path.join(temp_dir, input_filename)

            # Generate output filename with enhancement metadata
            name, ext = os.path.splitext(input_filename)
            file_type = get_file_type(input_filename)
            
            if file_type == 'video':
                output_filename = f"{name}_{enhancer}_s{skip_frames}_b{int(blend*100)}_enhanced.mp4"
            else:
                output_filename = f"{name}_{enhancer}_b{int(blend*100)}_enhanced{ext}"
                
            output_path = os.path.join(temp_dir, output_filename)

            # Enhanced download phase with multiple fallback methods
            logger.info("📥 Starting file download with enhanced retry logic...")
            download_start = time.time()
            download_success = False
            download_method = "unknown"

            # Method 1: Primary enhanced requests download
            logger.info("🔄 Trying primary download method (requests with retry)...")
            if download_input_file(input_url, input_path, max_retries=5):
                download_success = True
                download_method = "requests_enhanced"
            
            # Method 2: Fallback to cURL
            if not download_success:
                logger.warning("🔄 Primary download failed, trying cURL fallback...")
                if download_with_curl(input_url, input_path):
                    download_success = True
                    download_method = "curl"
            
            # Method 3: Final fallback to wget
            if not download_success:
                logger.warning("🔄 cURL failed, trying wget fallback...")
                if download_with_wget(input_url, input_path):
                    download_success = True
                    download_method = "wget"

            if not download_success:
                logger.error("❌ All download methods failed")
                return {
                    "error": "Failed to download input file after trying multiple methods (requests, cURL, wget)",
                    "status": "download_failed",
                    "job_id": job_id,
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "troubleshooting": {
                        "issue": "Network connectivity or server availability",
                        "url_tested": input_url,
                        "methods_tried": ["requests_enhanced", "curl", "wget"],
                        "suggestions": [
                            "Check if the URL is publicly accessible",
                            "Verify the file hasn't been moved or deleted",
                            "Try uploading to a different CDN or storage service",
                            "Contact support if the issue persists"
                        ]
                    },
                    "system_info": system_info
                }
                
            download_time = time.time() - download_start

            # File analysis and validation
            if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                return {
                    "error": "Downloaded file is empty or corrupted",
                    "status": "file_validation_failed",
                    "job_id": job_id,
                    "processing_time_seconds": round(time.time() - start_time, 2)
                }

            input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            actual_file_type = get_file_type(input_path)
            
            logger.info(f"📊 Input analysis: {actual_file_type}, {input_size_mb:.1f}MB")
            logger.info(f"✅ Download completed via {download_method} in {download_time:.1f}s")

            # Enhancement phase
            logger.info("🎨 Starting face enhancement process...")
            enhancement_start = time.time()
            
            success, message, enhancement_stats = run_face_enhancer(
                input_path=input_path,
                output_path=output_path,
                enhancer=enhancer,
                device=device,
                blend=blend,
                codeformer_w=codeformer_w,
                skip_frames=skip_frames
            )
            
            enhancement_time = time.time() - enhancement_start
            enhancement_stats['enhancement_time'] = round(enhancement_time, 2)
            enhancement_stats['download_method'] = download_method

            if not success:
                logger.error(f"❌ Enhancement failed: {message}")
                return {
                    "error": f"Face enhancement failed: {message}",
                    "status": "enhancement_failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "enhancement_stats": enhancement_stats,
                    "job_id": job_id,
                    "file_info": {
                        "input_size_mb": round(input_size_mb, 2),
                        "input_type": actual_file_type,
                        "download_method": download_method,
                        "download_time_seconds": round(download_time, 2)
                    },
                    "system_info": system_info
                }

            # Upload phase with retry logic
            logger.info("📤 Starting result upload...")
            upload_start = time.time()
            timestamp = int(time.time())
            output_object_name = f"enhanced/{enhancer}/{timestamp}/{job_id}_{output_filename}"
            
            try:
                output_url = upload_to_minio(output_path, output_object_name, max_retries=3)
            except Exception as e:
                logger.error(f"❌ Upload failed: {e}")
                return {
                    "error": f"Failed to upload result: {str(e)}",
                    "status": "upload_failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "enhancement_stats": enhancement_stats,
                    "job_id": job_id,
                    "system_info": system_info
                }
            
            upload_time = time.time() - upload_start

            # Final statistics and comprehensive success response
            total_time = time.time() - start_time
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            final_system_info = get_system_info()

            logger.info(f"✅ Job {job_id} completed successfully!")
            logger.info(f"⏱️ Timing breakdown: download={download_time:.1f}s, enhancement={enhancement_time:.1f}s, upload={upload_time:.1f}s, total={total_time:.1f}s")
            logger.info(f"📊 File sizes: input={input_size_mb:.1f}MB → output={output_size_mb:.1f}MB (ratio={output_size_mb/input_size_mb:.2f}x)")
            logger.info(f"🔧 Method: {download_method} download, {enhancer} enhancement")

            return {
                "output_url": output_url,
                "status": "completed",
                "job_id": job_id,
                "processing_time_seconds": round(total_time, 2),
                "timing": {
                    "download_seconds": round(download_time, 2),
                    "enhancement_seconds": round(enhancement_time, 2),
                    "upload_seconds": round(upload_time, 2),
                    "total_seconds": round(total_time, 2)
                },
                "file_info": {
                    "type": actual_file_type,
                    "input_size_mb": round(input_size_mb, 2),
                    "output_size_mb": round(output_size_mb, 2),
                    "size_ratio": round(output_size_mb / input_size_mb, 2) if input_size_mb > 0 else 1.0,
                    "input_filename": input_filename,
                    "output_filename": output_filename,
                    "download_method": download_method
                },
                "enhancement_config": {
                    "enhancer": enhancer,
                    "enhancer_info": SUPPORTED_ENHANCERS[enhancer],
                    "device": device,
                    "parameters": {
                        "blend_factor": blend,
                        "codeformer_w": codeformer_w if enhancer == 'codeformer' else None,
                        "skip_frames": skip_frames if actual_file_type == 'video' else None
                    }
                },
                "enhancement_stats": enhancement_stats,
                "system_info": {
                    "initial": system_info,
                    "final": final_system_info
                },
                "performance": {
                    "processing_speed_mbps": round(input_size_mb / enhancement_time, 2) if enhancement_time > 0 else 0,
                    "total_throughput_mbps": round(input_size_mb / total_time, 2) if total_time > 0 else 0,
                    "efficiency_score": round((output_size_mb / input_size_mb) / (total_time / 60), 2) if total_time > 0 else 0
                }
            }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Handler error for job {job_id}: {error_msg}")
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        return {
            "error": f"Internal service error: {error_msg}",
            "status": "internal_error",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id,
            "system_info": get_system_info(),
            "debug_info": {
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else "Enable debug logging for full traceback"
            }
        }

def health_check() -> Tuple[bool, str, Dict[str, Any]]:
    """Comprehensive health check with detailed system validation and diagnostics"""
    try:
        health_info = {
            "timestamp": time.time(),
            "version": "2.0",
            "system": get_system_info()
        }
        
        # Check core dependencies with version info
        try:
            import torch
            import cv2
            import onnxruntime
            import insightface
            import runpod
            
            health_info["dependencies"] = {
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "onnxruntime": onnxruntime.__version__,
                "onnxruntime_providers": onnxruntime.get_available_providers(),
                "insightface": insightface.__version__,
                "runpod": runpod.__version__
            }
            
            logger.info("✅ All dependencies available")
        except ImportError as e:
            return False, f"Missing dependency: {e}", health_info

        # Enhanced CUDA check
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            health_info["cuda"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
                "memory_cached_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
                "compute_capability": torch.cuda.get_device_capability(0)
            }
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)} ({health_info['cuda']['memory_total_gb']:.1f}GB)")
        else:
            health_info["cuda"] = {"available": False}
            logger.warning("⚠️ CUDA not available, will use CPU")

        # Comprehensive model verification
        models_ok, missing, model_sizes = verify_models()
        health_info["models"] = {
            "verified": models_ok,
            "total_count": len(MODEL_PATHS),
            "verified_count": len(model_sizes),
            "missing_count": len(missing),
            "missing_models": missing,
            "model_sizes_mb": model_sizes,
            "total_size_mb": round(sum(model_sizes.values()), 1) if model_sizes else 0,
            "model_paths": MODEL_PATHS
        }
        
        if not models_ok:
            return False, f"Missing {len(missing)}/{len(MODEL_PATHS)} models", health_info

        # Enhanced MinIO connection test
        if minio_client:
            try:
                # Test bucket access
                bucket_exists = minio_client.bucket_exists(MINIO_BUCKET)
                
                # Test upload/download capability with a small test file
                test_object = f"health_check_{int(time.time())}.txt"
                test_content = f"Health check at {time.time()}"
                
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(test_content)
                    test_file_path = f.name
                
                try:
                    # Test upload
                    minio_client.fput_object(MINIO_BUCKET, test_object, test_file_path)
                    
                    # Test download
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        download_path = f.name
                    minio_client.fget_object(MINIO_BUCKET, test_object, download_path)
                    
                    # Verify content
                    with open(download_path, 'r') as f:
                        downloaded_content = f.read()
                    
                    upload_success = (downloaded_content == test_content)
                    
                    # Cleanup test files
                    minio_client.remove_object(MINIO_BUCKET, test_object)
                    os.unlink(test_file_path)
                    os.unlink(download_path)
                    
                    health_info["storage"] = {
                        "minio_connected": True,
                        "bucket_exists": bucket_exists,
                        "upload_test": upload_success,
                        "endpoint": MINIO_ENDPOINT,
                        "bucket": MINIO_BUCKET
                    }
                    
                except Exception as e:
                    health_info["storage"] = {
                        "minio_connected": True,
                        "bucket_exists": bucket_exists,
                        "upload_test": False,
                        "error": str(e)
                    }
                    logger.warning(f"⚠️ MinIO upload test failed: {e}")
                    
                logger.info("✅ MinIO storage verified")
                
            except Exception as e:
                health_info["storage"] = {"minio_connected": False, "error": str(e)}
                return False, f"MinIO connection failed: {e}", health_info
        else:
            health_info["storage"] = {"minio_connected": False, "error": "Client not initialized"}
            return False, "MinIO not available", health_info

        # Check CLI tool and required files
        required_files = [
            '/app/face_enhancer_cli.py',
            '/app/face_enhancer_handler.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            elif not os.access(file_path, os.R_OK):
                missing_files.append(f"{file_path} (not readable)")
                
        if missing_files:
            return False, f"Missing required files: {missing_files}", health_info

        # Check required directories and permissions
        required_dirs = ['/app/enhancers', '/app/utils', '/tmp']
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                return False, f"Required directory missing: {dir_path}", health_info
            if not os.access(dir_path, os.R_OK | os.W_OK):
                return False, f"Insufficient permissions for: {dir_path}", health_info

        # Test basic CLI functionality (dry run)
        try:
            result = subprocess.run(
                ['python', '/app/face_enhancer_cli.py', '--list_models'],
                capture_output=True,
                text=True,
                timeout=30
            )
            health_info["cli_test"] = {
                "available": result.returncode == 0,
                "output_lines": len(result.stdout.split('\n')) if result.stdout else 0
            }
        except Exception as e:
            health_info["cli_test"] = {"available": False, "error": str(e)}
            logger.warning(f"⚠️ CLI test failed: {e}")

        # Final health summary
        health_summary = {
            "models_ready": models_ok,
            "storage_ready": health_info["storage"].get("minio_connected", False),
            "gpu_ready": cuda_available,
            "dependencies_ready": True,
            "total_model_size_mb": sum(model_sizes.values()) if model_sizes else 0,
            "available_enhancers": list(SUPPORTED_ENHANCERS.keys())
        }
        
        health_info["summary"] = health_summary

        logger.info("✅ Comprehensive health check passed")
        logger.info(f"📊 System ready: {len(SUPPORTED_ENHANCERS)} enhancers, {sum(model_sizes.values()):.1f}MB models")
        
        return True, "All systems operational and verified", health_info

    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return False, f"Health check failed: {str(e)}", {"error": str(e), "traceback": traceback.format_exc()}

# Main execution with enhanced startup sequence
if __name__ == "__main__":
    logger.info("🚀 Starting Face Enhancement Serverless Worker (Production v2.0)...")
    
    try:
        # Startup dependency verification with detailed logging
        logger.info("🔍 Verifying startup dependencies...")
        
        import torch
        import cv2
        import onnxruntime
        import insightface
        import runpod

        # Log comprehensive system information
        logger.info("🔥 System Information:")
        logger.info(f"   PyTorch: {torch.__version__}")
        logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            logger.info(f"   CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
            
        logger.info(f"   OpenCV: {cv2.__version__}")
        logger.info(f"   ONNX Runtime: {onnxruntime.__version__}")
        logger.info(f"   ONNX Providers: {onnxruntime.get_available_providers()}")
        logger.info(f"   InsightFace: {insightface.__version__}")
        logger.info(f"   RunPod SDK: {runpod.__version__}")
        
        # System resource information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        logger.info(f"   Memory: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB available")
        logger.info(f"   Disk Space: {disk.free / 1e9:.1f}GB free")
        logger.info(f"   CPU Cores: {psutil.cpu_count()}")

        # Comprehensive startup health check
        logger.info("🏥 Running comprehensive startup health check...")
        health_ok, health_msg, health_info = health_check()
        
        if not health_ok:
            logger.error(f"❌ Startup health check failed: {health_msg}")
            logger.error(f"🔍 Health details: {json.dumps(health_info, indent=2)}")
            sys.exit(1)

        # Success logging with service readiness confirmation
        logger.info(f"✅ Health check passed: {health_msg}")
        logger.info("🎨 Face Enhancement Service Ready for Production!")
        
        # Service capability summary
        models_info = health_info.get('models', {})
        storage_info = health_info.get('storage', {})
        
        logger.info(f"🔧 Service Capabilities:")
        logger.info(f"   Supported enhancers: {', '.join(SUPPORTED_ENHANCERS.keys())}")
        logger.info(f"   Total models: {models_info.get('verified_count', 0)}/{models_info.get('total_count', 0)}")
        logger.info(f"   Model storage: {models_info.get('total_size_mb', 0):.1f}MB")
        logger.info(f"   Storage backend: MinIO ({storage_info.get('endpoint', 'unknown')})")
        logger.info(f"   GPU acceleration: {'✅' if health_info.get('cuda', {}).get('available', False) else '❌'}")

        # Start RunPod serverless worker with error handling
        logger.info("🌐 Starting RunPod serverless worker...")
        runpod.serverless.start({"handler": handler})

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
        
        # Attempt to provide helpful error context
        if "torch" in str(e).lower():
            logger.error("💡 Suggestion: PyTorch installation issue - check CUDA compatibility")
        elif "onnx" in str(e).lower():
            logger.error("💡 Suggestion: ONNX Runtime issue - verify GPU provider availability")
        elif "model" in str(e).lower():
            logger.error("💡 Suggestion: Model loading issue - check model files and paths")
        elif "minio" in str(e).lower():
            logger.error("💡 Suggestion: Storage connection issue - verify MinIO configuration")
            
        sys.exit(1)
