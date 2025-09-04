#!/usr/bin/env python3
"""
Face Enhancement Serverless Handler (RunPod)
- Tải input từ URL (retry bền vững + cURL fallback)
- Gọi face_enhancer_cli.py qua subprocess
- Upload kết quả lên MinIO
- Health check đầy đủ khi khởi động

Yêu cầu:
- Đã build image có kèm face_enhancer_cli.py và các model ONNX vào vị trí tương ứng
- Biến môi trường MinIO đã cấu hình (xem bên dưới)
"""

import os
import sys
import time
import uuid
import json
import tempfile
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Tuple, List, Optional
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from minio import Minio
import runpod

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("face_enhancer_handler")

# ----------------------------
# Config (environment-driven)
# ----------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "media.aiclip.ai")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "VtZ6MUPfyTOH3qSiohA2")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "8boVPVIynLEKcgXirrcePxvjSk7gReIDD9pwto3t")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "video")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")

# CLI path
CLI_PATH = os.environ.get("FACE_ENHANCER_CLI", "/app/face_enhancer_cli.py")

# Timeout (seconds)
ENHANCER_TIMEOUT = int(os.environ.get("ENHANCER_TIMEOUT", "1800"))  # 30 minutes

# Supported enhancers
SUPPORTED_ENHANCERS = ['gfpgan', 'gpen', 'codeformer', 'restoreformer16', 'restoreformer32']

# Model paths
MODEL_PATHS = {
    'gfpgan': '/app/enhancers/GFPGAN/GFPGANv1.4.onnx',
    'gpen': '/app/enhancers/GPEN/GPEN-BFR-512.onnx',
    'codeformer': '/app/enhancers/Codeformer/codeformer.onnx',
    'restoreformer16': '/app/enhancers/restoreformer/restoreformer16.onnx',
    'restoreformer32': '/app/enhancers/restoreformer/restoreformer32.onnx',
    'retinaface': '/app/utils/scrfd_2.5g_bnkps.onnx',
}

# ----------------------------
# MinIO client (lazy init)
# ----------------------------
_minio_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    global _minio_client
    if _minio_client is None:
        if not (MINIO_ENDPOINT and MINIO_ACCESS_KEY and MINIO_SECRET_KEY and MINIO_BUCKET):
            raise RuntimeError("MinIO config missing (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET)")
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        logger.info(f"MinIO initialized -> endpoint={MINIO_ENDPOINT}, bucket={MINIO_BUCKET}, secure={MINIO_SECURE}")
    return _minio_client

# ----------------------------
# Utilities
# ----------------------------


def get_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[15].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
        return 'video'
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        return 'image'
    return 'unknown'


def _retry_kwargs_for_urllib3() -> dict:
    # Tương thích urllib3 v1/v2: ưu tiên allowed_methods, fallback method_whitelist
    # https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html [v2]
    # https://urllib3.readthedocs.io/en/1.26.x/reference/urllib3.util.html [v1]
    params = Retry.__init__.__code__.co_varnames
    if "allowed_methods" in params:
        return {"allowed_methods": frozenset(["HEAD", "GET", "OPTIONS"])}
    else:
        return {"method_whitelist": frozenset(["HEAD", "GET", "OPTIONS"])}


def _make_retry_session(total: int = 5, backoff: float = 1.0) -> requests.Session:
    retry = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
        **_retry_kwargs_for_urllib3()
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_input_file(url: str, output_path: str) -> bool:
    """
    Tải file với retry bền vững; nếu thất bại, fallback sang cURL (không dùng --buffer-size vì không phải option của curl CLI).
    """
    try:
        # HEAD kiểm tra nhẹ, nếu fail thì bỏ qua (nhiều server chặn HEAD)
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            if head.status_code >= 400:
                # Probe GET nhẹ để xác thực tồn tại
                probe = requests.get(url, timeout=10, stream=True)
                probe.raise_for_status()
        except Exception:
            pass

        # GET streaming với retry
        session = _make_retry_session(total=5, backoff=1.0)
        with session.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Downloaded input: {size_mb:.1f}MB -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Primary download failed: {e}")

        # Fallback: cURL tối giản (không dùng --buffer-size; đó là libcurl C API, không phải tham số CLI)
        try:
            curl_cmd = [
                "curl", "-L", "-C", "-",
                "--max-time", "600",
                "--connect-timeout", "30",
                "--retry", "3",
                "--retry-delay", "2",
                "--retry-max-time", "300",
                "--fail", "-sS",
                "-o", output_path,
                url
            ]
            logger.info(f"Trying cURL fallback: {' '.join(curl_cmd)}")
            res = subprocess.run(curl_cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"cURL fallback succeeded: {size_mb:.1f}MB")
                return True
            logger.error(f"cURL failed rc={res.returncode}, stderr={res.stderr[:200]}")
        except Exception as ee:
            logger.error(f"cURL error: {ee}")

        return False


def upload_to_minio(local_path: str, object_name: str) -> str:
    client = get_minio_client()
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Ensure bucket exists
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            logger.info(f"Created bucket: {MINIO_BUCKET}")
    except Exception as e:
        logger.warning(f"Bucket check/create warning: {e}")

    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info(f"Uploading to MinIO: {object_name} ({size_mb:.1f}MB)")
    client.fput_object(MINIO_BUCKET, object_name, local_path)
    url = f"https://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}"
    logger.info(f"Uploaded -> {url}")
    return url


def verify_models() -> Tuple[bool, List[str]]:
    logger.info("Verifying enhancement models...")
    missing, total_size = [], 0.0
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                sz = os.path.getsize(path) / (1024 * 1024)
                total_size += sz
                logger.info(f"✓ {name}: {sz:.1f}MB")
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
                missing.append(f"{name}: {path} (read error)")
        else:
            logger.error(f"Missing model: {name} -> {path}")
            missing.append(f"{name}: {path}")
    if missing:
        logger.error(f"Missing {len(missing)}/{len(MODEL_PATHS)} models")
        return False, missing
    logger.info(f"All models verified, total size: {total_size:.1f}MB")
    return True, []


def run_face_enhancer(
    input_path: str,
    output_path: str,
    enhancer: str,
    device: str = 'cuda',
    blend: float = 0.8,
    codeformer_w: float = 0.9,
    skip_frames: int = 1
) -> Tuple[bool, str]:
    """
    Gọi CLI enhancer qua subprocess với timeout, hỗ trợ tham số video/image.
    """
    cmd = [
        sys.executable, CLI_PATH,
        "--input", input_path,
        "--output", output_path,
        "--enhancer", enhancer,
        "--device", device,
        "--blend", str(blend),
        "--server_mode",
        "--quiet"
    ]

    if enhancer == "codeformer":
        cmd.extend(["--codeformer_w", str(codeformer_w)])

    if get_file_type(input_path) == "video" and skip_frames > 1:
        cmd.extend(["--skip_frames", str(skip_frames)])

    logger.info(f"Launching enhancer: {' '.join(cmd)}")
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=ENHANCER_TIMEOUT, check=False
        )
        elapsed = time.time() - start

        if proc.stdout:
            logger.debug(f"enhancer stdout: {proc.stdout[:500]}")
        if proc.stderr:
            if proc.returncode != 0:
                logger.error(f"enhancer stderr: {proc.stderr[:500]}")
            else:
                logger.info(f"enhancer info: {proc.stderr[:200]}")

        if proc.returncode == 0 and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Enhancement OK in {elapsed:.1f}s, output {size_mb:.1f}MB")
            return True, f"Enhancement completed in {elapsed:.1f}s"
        else:
            msg = f"Enhancer failed rc={proc.returncode}, output missing={not os.path.exists(output_path)}"
            return False, msg

    except subprocess.TimeoutExpired:
        logger.error("Enhancer timed out")
        return False, "Processing timed out"
    except Exception as e:
        logger.error(f"Enhancer error: {e}")
        return False, str(e)


def validate_input_parameters(job_input: dict) -> Tuple[bool, str]:
    try:
        if "input_url" not in job_input or not job_input["input_url"]:
            return False, "Missing required parameter: input_url"
        if "enhancer" not in job_input or not job_input["enhancer"]:
            return False, "Missing required parameter: enhancer"
        enhancer = job_input["enhancer"]
        if enhancer not in SUPPORTED_ENHANCERS:
            return False, f"Unsupported enhancer. Must be one of: {', '.join(SUPPORTED_ENHANCERS)}"

        device = job_input.get("device", "cuda")
        if device not in ("cuda", "cpu"):
            return False, "device must be 'cuda' or 'cpu'"

        blend = float(job_input.get("blend", 0.8))
        if not 0.0 <= blend <= 1.0:
            return False, "blend must be in [0.0, 1.0]"

        if enhancer == "codeformer":
            codeformer_w = float(job_input.get("codeformer_w", 0.9))
            if not 0.0 <= codeformer_w <= 1.0:
                return False, "codeformer_w must be in [0.0, 1.0]"

        skip_frames = int(job_input.get("skip_frames", 1))
        if not 1 <= skip_frames <= 5:
            return False, "skip_frames must be in [1, 5]"

        # URL basic check (không fail cứng nếu HEAD bị chặn)
        try:
            r = requests.head(job_input["input_url"], timeout=10, allow_redirects=True)
            if r.status_code >= 400:
                # fallback probe GET
                g = requests.get(job_input["input_url"], timeout=10, stream=True)
                g.raise_for_status()
        except Exception:
            # vẫn cho qua, chặn ở bước download chính
            pass

        return True, "Valid"
    except Exception as e:
        return False, f"Parameter validation error: {e}"

# ----------------------------
# RunPod handler
# ----------------------------


def handler(job):
    job_id = job.get("id", f"job_{uuid.uuid4().hex[:8]}")
    start_time = time.time()
    try:
        job_input = job.get("input", {}) or {}
        ok, msg = validate_input_parameters(job_input)
        if not ok:
            return {"error": msg, "status": "failed", "job_id": job_id}

        input_url = job_input["input_url"]
        enhancer = job_input["enhancer"]
        device = job_input.get("device", "cuda")
        blend = float(job_input.get("blend", 0.8))
        codeformer_w = float(job_input.get("codeformer_w", 0.9))
        skip_frames = int(job_input.get("skip_frames", 1))

        # Verify models
        models_ok, missing = verify_models()
        if not models_ok:
            return {
                "error": "Required models are missing",
                "missing_models": missing,
                "status": "failed",
                "job_id": job_id
            }

        # Workdir
        with tempfile.TemporaryDirectory(prefix=f"face_enhance_{job_id}_") as tmp:
            parsed = urlparse(input_url)
            input_filename = os.path.basename(parsed.path) or "input_file.mp4"
            input_path = os.path.join(tmp, input_filename)

            # Output filename
            name, ext = os.path.splitext(input_filename)
            ftype = get_file_type(input_filename)
            if ftype == "video":
                output_filename = f"{name}_enhanced.mp4"
            elif ftype == "image":
                output_filename = f"{name}_enhanced{ext}"
            else:
                # default to mp4
                output_filename = f"{name}_enhanced.mp4"
                ftype = "video"
            output_path = os.path.join(tmp, output_filename)

            # Download
            logger.info(f"Downloading: {input_url}")
            if not download_input_file(input_url, input_path):
                return {"error": "Failed to download input file", "status": "failed", "job_id": job_id}

            input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            logger.info(f"Input: type={ftype}, size={input_size_mb:.1f}MB")

            # Enhance
            enh_start = time.time()
            ok, emsg = run_face_enhancer(
                input_path=input_path,
                output_path=output_path,
                enhancer=enhancer,
                device=device,
                blend=blend,
                codeformer_w=codeformer_w,
                skip_frames=skip_frames
            )
            enh_time = time.time() - enh_start
            if not ok:
                return {
                    "error": f"Enhancement failed: {emsg}",
                    "status": "failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "job_id": job_id
                }

            # Upload
            object_name = f"face_enhanced_{job_id}_{uuid.uuid4().hex[:8]}_{output_filename}"
            try:
                output_url = upload_to_minio(output_path, object_name)
            except Exception as e:
                return {
                    "error": f"Failed to upload result: {e}",
                    "status": "failed",
                    "processing_time_seconds": round(time.time() - start_time, 2),
                    "job_id": job_id
                }

            total_time = time.time() - start_time
            out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            return {
                "output_url": output_url,
                "processing_time_seconds": round(total_time, 2),
                "enhancement_time_seconds": round(enh_time, 2),
                "file_info": {
                    "type": ftype,
                    "input_size_mb": round(input_size_mb, 2),
                    "output_size_mb": round(out_size_mb, 2),
                    "enhancement_ratio": round(out_size_mb / input_size_mb, 2) if input_size_mb > 0 else 1.0
                },
                "enhancement_params": {
                    "enhancer": enhancer,
                    "device": device,
                    "blend_factor": blend,
                    "codeformer_w": codeformer_w if enhancer == "codeformer" else None,
                    "skip_frames": skip_frames if ftype == "video" else None
                },
                "status": "completed",
                "job_id": job_id
            }

    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "status": "failed",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "job_id": job_id
        }

# ----------------------------
# Health check and startup
# ----------------------------


def health_check() -> Tuple[bool, str]:
    try:
        # Basic libs
        import torch
        import cv2
        import onnxruntime

        # CUDA
        if torch.cuda.is_available():
            _ = torch.cuda.get_device_name(0)

        # Models
        ok, missing = verify_models()
        if not ok:
            return False, f"Missing models: {len(missing)}"

        # MinIO client + bucket
        client = get_minio_client()
        try:
            client.bucket_exists(MINIO_BUCKET)
        except Exception as e:
            return False, f"MinIO bucket check failed: {e}"

        # CLI tool
        if not os.path.exists(CLI_PATH):
            return False, f"CLI not found at {CLI_PATH}"

        return True, "All systems operational"
    except Exception as e:
        return False, f"Health check failed: {e}"


if __name__ == "__main__":
    try:
        # System info (best-effort)
        try:
            import torch
            logger.info(f"PyTorch: {getattr(torch, '__version__', 'unknown')}")
            logger.info(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        except Exception:
            pass

        try:
            import cv2
            logger.info(f"OpenCV: {getattr(cv2, '__version__', 'unknown')}")
        except Exception:
            pass

        try:
            import onnxruntime
            logger.info(f"ONNX Runtime: {getattr(onnxruntime, '__version__', 'unknown')}")
        except Exception:
            pass

        ok, msg = health_check()
        if not ok:
            logger.error(f"Health check failed: {msg}")
            sys.exit(1)
        logger.info(f"Health check passed: {msg}")

        # Start serverless worker
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

