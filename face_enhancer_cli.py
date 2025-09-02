#!/usr/bin/env python3
"""
Video Face Enhancer CLI Tool - Server Optimized
Công cụ nâng cao chất lượng khuôn mặt trong video tối ưu cho Linux server

Sử dụng:
    python face_enhancer_cli.py --input video.mp4 --output enhanced.mp4 --enhancer gfpgan
    python face_enhancer_cli.py --input_dir ./videos --output_dir ./enhanced --enhancer codeformer --batch

Các enhancer hỗ trợ:
    - gfpgan: General-purpose face enhancement
    - gpen: GAN-based enhancement  
    - codeformer: Transformer-based restoration
    - restoreformer: Advanced restoration
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import glob
import subprocess
import platform
import logging

# Setup server-friendly logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenCV import with fallback for headless servers
try:
    import cv2
    # Disable GUI features for server environment
    if 'DISPLAY' not in os.environ:
        os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
    CV2_AVAILABLE = True
except ImportError:
    logger.error("OpenCV not available. Install with: pip install opencv-python-headless")
    CV2_AVAILABLE = False
    sys.exit(1)

# Import face detection và alignment
try:
    from utils.retinaface import RetinaFace
    from utils.face_alignment import get_cropped_head_256
    FACE_DETECTION_AVAILABLE = True
    logger.info("RetinaFace detection available")
except ImportError:
    logger.warning("RetinaFace modules not available. Using OpenCV face detection fallback.")
    FACE_DETECTION_AVAILABLE = False

# Import các enhancer classes
try:
    from enhancers.GFPGAN.GFPGAN import GFPGAN
    from enhancers.GPEN.GPEN import GPEN
    from enhancers.Codeformer.Codeformer import CodeFormer
    from enhancers.restoreformer.restoreformer16 import RestoreFormer as RestoreFormer16
    from enhancers.restoreformer.restoreformer32 import RestoreFormer as RestoreFormer32
    ENHANCERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Enhancer modules not available: {e}")
    ENHANCERS_AVAILABLE = False
    sys.exit(1)

class VideoFaceEnhancer:
    """Video Face Enhancement optimized for server environment"""
    
    def __init__(self, device='cpu', server_mode=True):
        self.device = device
        self.server_mode = server_mode
        self.enhancers = {}
        self.model_paths = {
            'gfpgan': 'enhancers/GFPGAN/GFPGANv1.4.onnx',
            'gpen': 'enhancers/GPEN/GPEN-BFR-512.onnx', 
            'codeformer': 'enhancers/Codeformer/codeformer.onnx',
            'restoreformer16': 'enhancers/restoreformer/restoreformer16.onnx',
            'restoreformer32': 'enhancers/restoreformer/restoreformer32.onnx'
        }
        
        # Initialize face detector nếu có
        if FACE_DETECTION_AVAILABLE:
            try:
                provider_opts = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"] if device == 'cuda' else ["CPUExecutionProvider"]
                self.face_detector = RetinaFace("utils/scrfd_2.5g_bnkps.onnx", provider=provider_opts)
                logger.info("RetinaFace detector loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load RetinaFace: {e}. Using OpenCV fallback.")
                self.face_detector = None
        else:
            self.face_detector = None
        
    def load_enhancer(self, enhancer_name):
        """Load enhancer model theo tên"""
        if enhancer_name in self.enhancers:
            return self.enhancers[enhancer_name]
            
        model_path = self.model_paths.get(enhancer_name)
        if not model_path or not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            logger.info("Download models from: https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ")
            return None
            
        try:
            if enhancer_name == 'gfpgan':
                enhancer = GFPGAN(model_path, self.device)
            elif enhancer_name == 'gpen':
                enhancer = GPEN(model_path, self.device)
            elif enhancer_name == 'codeformer':
                enhancer = CodeFormer(model_path, self.device)
            elif enhancer_name == 'restoreformer16':
                enhancer = RestoreFormer16(model_path, self.device)
            elif enhancer_name == 'restoreformer32':
                enhancer = RestoreFormer32(model_path, self.device)
            else:
                logger.error(f"Unsupported enhancer: {enhancer_name}")
                return None
                
            self.enhancers[enhancer_name] = enhancer
            logger.info(f"Loaded {enhancer_name} enhancer successfully")
            return enhancer
            
        except Exception as e:
            logger.error(f"Failed to load {enhancer_name}: {str(e)}")
            return None

    def detect_and_align_face(self, frame, target_size=256):
        """Detect và align khuôn mặt trong frame"""
        if not self.face_detector:
            # Fallback: sử dụng OpenCV face detection (server-friendly)
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]  # Lấy face đầu tiên
                    face_crop = frame[y:y+h, x:x+w]
                    face_crop = cv2.resize(face_crop, (target_size, target_size))
                    
                    # Tạo transform matrix đơn giản
                    scale_x = w / target_size
                    scale_y = h / target_size
                    transform_matrix = np.array([[scale_x, 0, x], [0, scale_y, y]], dtype=np.float32)
                    
                    return face_crop, transform_matrix, (x, y, w, h)
                else:
                    return None, None, None
            except Exception as e:
                logger.warning(f"OpenCV face detection failed: {e}")
                return None, None, None
        
        try:
            # Sử dụng RetinaFace detection
            bboxes, kpss = self.face_detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
            
            # Kiểm tra xem có detection nào không
            if bboxes is not None and len(bboxes) > 0 and kpss is not None and len(kpss) > 0:
                # Lấy face đầu tiên
                bbox = bboxes[0]  # bbox format: [x1, y1, x2, y2, confidence]
                kps = kpss[0]     # keypoints shape: (5, 2) for 5 facial landmarks
                
                # Reshape keypoints nếu cần (từ (5,2) thành flat array cho compatibility)
                if kps.shape == (5, 2):
                    kps_flat = kps.flatten()  # Convert to [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
                else:
                    kps_flat = kps
                
                aligned_face, transform_matrix = get_cropped_head_256(frame, kps, size=target_size, scale=1.0)
                
                # Sử dụng bbox từ detection thay vì tính từ keypoints
                x, y, x2, y2 = bbox[:4]
                x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
                w, h = x2 - x, y2 - y
                
                return aligned_face, transform_matrix, (x, y, w, h)
            else:
                return None, None, None
                
        except Exception as e:
            logger.warning(f"Face detection error: {e}")
            return None, None, None

    def enhance_face(self, face_image, enhancer_name, blend_factor=1.0, codeformer_w=0.9):
        """Enhance khuôn mặt đã được align"""
        enhancer = self.load_enhancer(enhancer_name)
        if enhancer is None:
            return None
            
        try:
            # Lưu original size để resize output về cùng size
            original_shape = face_image.shape
            
            if enhancer_name == 'codeformer':
                enhanced = enhancer.enhance(face_image, w=codeformer_w)
            else:
                enhanced = enhancer.enhance(face_image)
            
            # Đảm bảo enhanced có cùng size với original
            if enhanced.shape != original_shape:
                logger.debug(f"Resizing enhanced from {enhanced.shape} to {original_shape}")
                enhanced = cv2.resize(enhanced, (original_shape[1], original_shape[0]))
                
            # Blend với ảnh gốc nếu blend_factor < 1.0
            if blend_factor < 1.0:
                # Đảm bảo cả hai ảnh có cùng shape và dtype
                enhanced_float = enhanced.astype(np.float32)
                face_image_float = face_image.astype(np.float32)
                
                # Double check shape compatibility (không nên xảy ra nữa)
                if enhanced_float.shape != face_image_float.shape:
                    logger.error(f"Unexpected shape mismatch: enhanced {enhanced_float.shape}, original {face_image_float.shape}")
                    return enhanced.astype(np.uint8)
                
                enhanced = cv2.addWeighted(
                    enhanced_float, blend_factor,
                    face_image_float, 1.0 - blend_factor, 0.0
                ).astype(np.uint8)
                
            return enhanced
            
        except Exception as e:
            logger.error(f"Face enhancement failed: {str(e)}")
            return None

    def blend_face_to_frame(self, frame, enhanced_face, transform_matrix, bbox, blend_factor=0.9):
        """Blend enhanced face trở lại frame gốc"""
        if enhanced_face is None or transform_matrix is None:
            return frame
            
        try:
            h, w = frame.shape[:2]
            
            # Inverse transform để đưa face về vị trí gốc
            inv_transform = cv2.invertAffineTransform(transform_matrix)
            warped_face = cv2.warpAffine(enhanced_face, inv_transform, (w, h))
            
            # Ensure frame và warped_face có cùng shape và dtype
            frame = frame.astype(np.float32)
            warped_face = warped_face.astype(np.float32)
            
            # Kiểm tra shape compatibility
            if warped_face.shape != frame.shape:
                logger.warning(f"Shape mismatch: frame {frame.shape}, warped_face {warped_face.shape}")
                # Resize warped_face nếu cần
                if len(warped_face.shape) == 3 and len(frame.shape) == 3:
                    if warped_face.shape[:2] != frame.shape[:2]:
                        warped_face = cv2.resize(warped_face, (w, h))
                else:
                    return frame.astype(np.uint8)
            
            # Tạo mask cho blend
            x, y, face_w, face_h = bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (x + face_w//2, y + face_h//2), 
                       (face_w//2, face_h//2), 0, 0, 360, 255, -1)
            
            # Blur mask để blend mượt hơn
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            mask = mask.astype(np.float32) / 255.0
            
            # Đảm bảo mask có đúng số channels
            if len(frame.shape) == 3:
                mask = np.stack([mask] * frame.shape[2], axis=2)
            
            # Blend face với frame
            result = (mask * warped_face + (1 - mask) * frame).astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.warning(f"Face blending failed: {e}")
            return frame

def validate_paths(input_path, output_path, input_dir, output_dir):
    """Kiểm tra tính hợp lệ của các đường dẫn"""
    if input_path and not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
        
    if input_dir and not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return False
        
    if output_path:
        output_parent = Path(output_path).parent
        output_parent.mkdir(parents=True, exist_ok=True)
        
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    return True

def get_supported_video_files(directory):
    """Lấy danh sách file video được hỗ trợ từ thư mục"""
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    return sorted(files)

def get_supported_image_files(directory):
    """Lấy danh sách file ảnh được hỗ trợ từ thư mục"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
        files.extend(glob.glob(os.path.join(directory, ext.upper())))
    return sorted(files)

def process_video(enhancer, input_path, output_path, enhancer_name, 
                 blend_factor, codeformer_w, skip_frames=1, quality='medium'):
    """Xử lý video với face enhancement - Server optimized"""
    logger.info(f"Processing video: {input_path}")
    
    # Mở video input
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {input_path}")
        return False
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video info: {width}x{height}, {fps}fps, {total_frames} frames")
    logger.info(f"Processing every {skip_frames} frame(s)")
    
    # Setup video writer với quality settings
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_count = 0
    enhanced_count = 0
    start_time = time.time()
    
    # Process frames với server-optimized progress tracking
    logger.info("Starting frame processing...")
    with tqdm(total=total_frames, desc=f"Processing {enhancer_name}", unit="frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames để tăng tốc độ
            if frame_count % skip_frames == 0:
                # Detect và enhance face
                aligned_face, transform_matrix, bbox = enhancer.detect_and_align_face(frame)
                
                if aligned_face is not None:
                    enhanced_face = enhancer.enhance_face(
                        aligned_face, enhancer_name, blend_factor, codeformer_w
                    )
                    
                    if enhanced_face is not None:
                        frame = enhancer.blend_face_to_frame(
                            frame, enhanced_face, transform_matrix, bbox, blend_factor
                        )
                        enhanced_count += 1
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                logger.debug(f"Processed {frame_count}/{total_frames} frames, {fps_current:.1f} fps")
    
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
    logger.info(f"Enhanced {enhanced_count}/{frame_count} frames in {processing_time:.1f}s")
    
    # Merge với audio từ video gốc using FFmpeg
    try:
        logger.info("Merging audio with FFmpeg...")
        
        # Server-optimized FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-v', 'warning',  # Quiet output for server
            '-i', temp_output,
            '-i', input_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            os.remove(temp_output)
            logger.info(f"Video saved successfully: {output_path}")
            return True
        else:
            logger.warning(f"FFmpeg warning: {result.stderr}")
            # Fallback: rename temp file
            os.rename(temp_output, output_path)
            logger.info(f"Video saved (no audio): {output_path}")
            return True
            
    except Exception as e:
        logger.warning(f"Audio merge failed: {e}")
        try:
            os.rename(temp_output, output_path)
            logger.info(f"Video saved (no audio): {output_path}")
            return True
        except:
            logger.error("Failed to save video")
            return False

def process_image(enhancer, input_path, output_path, enhancer_name, 
                 blend_factor, codeformer_w):
    """Xử lý ảnh đơn lẻ với face enhancement"""
    logger.info(f"Processing image: {input_path}")
    
    # Đọc ảnh
    image = cv2.imread(input_path)
    if image is None:
        logger.error(f"Cannot read image: {input_path}")
        return False
    
    # Detect và enhance face
    aligned_face, transform_matrix, bbox = enhancer.detect_and_align_face(image)
    
    if aligned_face is not None:
        enhanced_face = enhancer.enhance_face(
            aligned_face, enhancer_name, blend_factor, codeformer_w
        )
        
        if enhanced_face is not None:
            result = enhancer.blend_face_to_frame(
                image, enhanced_face, transform_matrix, bbox, blend_factor
            )
            logger.info("Face enhancement completed")
        else:
            result = image
            logger.warning("Face enhancement failed, using original image")
    else:
        logger.warning("No face detected, keeping original image")
        result = image
    
    # Lưu ảnh
    success = cv2.imwrite(output_path, result)
    if success:
        logger.info(f"Image saved: {output_path}")
        return True
    else:
        logger.error(f"Failed to save image: {output_path}")
        return False

def process_batch_files(enhancer, input_dir, output_dir, enhancer_name,
                       blend_factor, codeformer_w, file_type='auto', 
                       skip_frames=1, quality='medium'):
    """Xử lý batch files từ thư mục - Server optimized"""
    
    if file_type == 'auto':
        video_files = get_supported_video_files(input_dir)
        image_files = get_supported_image_files(input_dir)
        input_files = video_files + image_files
    elif file_type == 'video':
        input_files = get_supported_video_files(input_dir)
    elif file_type == 'image':
        input_files = get_supported_image_files(input_dir)
    else:
        input_files = []
    
    if not input_files:
        logger.error(f"No supported files found in: {input_dir}")
        return 0
        
    logger.info(f"Found {len(input_files)} files to process")
    
    success_count = 0
    total_start_time = time.time()
    
    for i, input_file in enumerate(input_files, 1):
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        
        logger.info(f"Processing file {i}/{len(input_files)}: {filename}")
        
        # Determine output extension
        if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
            output_file = os.path.join(output_dir, f"{name}_{enhancer_name}.mp4")
            is_video = True
        else:
            output_file = os.path.join(output_dir, f"{name}_{enhancer_name}{ext}")
            is_video = False
        
        try:
            if is_video:
                success = process_video(
                    enhancer, input_file, output_file, enhancer_name,
                    blend_factor, codeformer_w, skip_frames, quality
                )
            else:
                success = process_image(
                    enhancer, input_file, output_file, enhancer_name,
                    blend_factor, codeformer_w
                )
            
            if success:
                success_count += 1
                logger.info(f"Successfully processed: {filename}")
            else:
                logger.error(f"Failed to process: {filename}")
                
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
    
    total_time = time.time() - total_start_time
    logger.info(f"Batch processing completed: {success_count}/{len(input_files)} files in {total_time:.1f}s")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description='Video Face Enhancer CLI - Server Optimized for Linux',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Server Usage Examples:
  # Basic video enhancement
  python face_enhancer_cli.py -i input.mp4 -o output.mp4 -e gfpgan --device cuda
  
  # Fast batch processing with skip frames
  python face_enhancer_cli.py --input_dir ./videos --output_dir ./enhanced -e restoreformer16 --batch --skip_frames 3 --device cuda
  
  # High quality single video
  python face_enhancer_cli.py -i video.mp4 -o enhanced.mp4 -e restoreformer32 --device cuda --verbose
  
  # Batch with specific file type
  python face_enhancer_cli.py --input_dir ./content --output_dir ./enhanced -e gfpgan --batch --file_type video --skip_frames 2

Enhancers (optimized for server):
  - gfpgan: General-purpose, balanced speed/quality
  - gpen: High detail, good for professional content  
  - codeformer: Smart restoration with identity control
  - restoreformer16: Fastest processing (recommended for batch)
  - restoreformer32: Best quality (slower)
        """
    )
    
    # Input/Output options
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('-i', '--input', type=str, help='Input video/image file')
    io_group.add_argument('-o', '--output', type=str, help='Output file path') 
    io_group.add_argument('--input_dir', type=str, help='Input directory for batch processing')
    io_group.add_argument('--output_dir', type=str, help='Output directory for batch processing')
    
    # Enhancement options
    enhance_group = parser.add_argument_group('Enhancement')
    enhance_group.add_argument('-e', '--enhancer', type=str, required=True,
                             choices=['gfpgan', 'gpen', 'codeformer', 'restoreformer16', 'restoreformer32'],
                             help='Enhancer model to use')
    enhance_group.add_argument('--blend', type=float, default=0.8, metavar='0.0-1.0',
                             help='Blend factor with original face (default: 0.8)')
    enhance_group.add_argument('--codeformer_w', type=float, default=0.9, metavar='0.0-1.0',
                             help='CodeFormer fidelity parameter (0.0=preserve identity, 1.0=best quality)')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing')
    proc_group.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                           help='Processing device (cpu or cuda)')
    proc_group.add_argument('--batch', action='store_true',
                           help='Enable batch processing mode')
    proc_group.add_argument('--file_type', type=str, default='auto', 
                           choices=['auto', 'video', 'image'],
                           help='File type filter for batch processing')
    
    # Performance options
    perf_group = parser.add_argument_group('Performance (Server Optimized)')
    perf_group.add_argument('--skip_frames', type=int, default=1, metavar='1-5',
                           help='Process every Nth frame (1=all frames, 2=every other frame, etc.)')
    perf_group.add_argument('--quality', type=str, default='medium',
                           choices=['fast', 'medium', 'high'],
                           help='Output quality preset')
    
    # Server options
    server_group = parser.add_argument_group('Server Options')
    server_group.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
    server_group.add_argument('--quiet', '-q', action='store_true',
                           help='Minimal output (errors only)')
    server_group.add_argument('--server_mode', action='store_true', default=True,
                           help='Server mode (no GUI dependencies)')
    
    # Utility options
    util_group = parser.add_argument_group('Utility')
    util_group.add_argument('--list_models', action='store_true',
                           help='List required model files and exit')
    util_group.add_argument('--check_deps', action='store_true',
                           help='Check dependencies and exit')
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Check dependencies
    if args.check_deps:
        logger.info("Checking dependencies...")
        logger.info(f"OpenCV: {'✓' if CV2_AVAILABLE else '✗'}")
        logger.info(f"Face Detection: {'✓' if FACE_DETECTION_AVAILABLE else '✗'}")
        logger.info(f"Enhancers: {'✓' if ENHANCERS_AVAILABLE else '✗'}")
        return
    
    # List models
    if args.list_models:
        logger.info("Required model files:")
        logger.info("GFPGAN: enhancers/GFPGAN/GFPGANv1.4.onnx")
        logger.info("GPEN: enhancers/GPEN/GPEN-BFR-512.onnx")
        logger.info("CodeFormer: enhancers/Codeformer/codeformer.onnx")
        logger.info("RestoreFormer: enhancers/restoreformer/restoreformer.onnx")
        logger.info("Face Detection: utils/scrfd_2.5g_bnkps.onnx")
        logger.info("\nDownload from: https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ")
        return
    
    # Validate arguments
    if args.batch:
        if not args.input_dir or not args.output_dir:
            logger.error("Batch mode requires --input_dir and --output_dir")
            return
    else:
        if not args.input or not args.output:
            logger.error("Single file mode requires --input and --output")
            return
    
    # Validate parameters
    if not 0.0 <= args.blend <= 1.0:
        logger.error("Blend factor must be between 0.0 and 1.0")
        return
        
    if not 0.0 <= args.codeformer_w <= 1.0:
        logger.error("CodeFormer w parameter must be between 0.0 and 1.0")
        return
        
    if not 1 <= args.skip_frames <= 5:
        logger.error("Skip frames must be between 1 and 5")
        return
    
    # Validate paths
    if not validate_paths(args.input, args.output, args.input_dir, args.output_dir):
        return
    
    # Server info logging
    logger.info(f"Video Face Enhancer CLI - Server Mode")
    logger.info(f"Enhancer: {args.enhancer.upper()}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Blend factor: {args.blend}")
    logger.info(f"Skip frames: {args.skip_frames} ({'Fast' if args.skip_frames > 1 else 'Quality'} mode)")
    if args.enhancer == 'codeformer':
        logger.info(f"CodeFormer w: {args.codeformer_w}")
    
    # Initialize enhancer with server mode
    enhancer = VideoFaceEnhancer(args.device, server_mode=args.server_mode)
    
    start_time = time.time()
    
    if args.batch:
        # Batch processing
        success_count = process_batch_files(
            enhancer, args.input_dir, args.output_dir, args.enhancer,
            args.blend, args.codeformer_w, args.file_type,
            args.skip_frames, args.quality
        )
        
        # Calculate totals for reporting
        if args.file_type == 'auto':
            video_files = get_supported_video_files(args.input_dir)
            image_files = get_supported_image_files(args.input_dir)
            total_files = len(video_files) + len(image_files)
        elif args.file_type == 'video':
            total_files = len(get_supported_video_files(args.input_dir))
        elif args.file_type == 'image':
            total_files = len(get_supported_image_files(args.input_dir))
        
        logger.info(f"Batch processing result: {success_count}/{total_files} files processed successfully")
        
    else:
        # Single file processing
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
            success = process_video(
                enhancer, args.input, args.output, args.enhancer,
                args.blend, args.codeformer_w,
                args.skip_frames, args.quality
            )
        else:
            success = process_image(
                enhancer, args.input, args.output, args.enhancer,
                args.blend, args.codeformer_w
            )
        
        if success:
            logger.info("Processing completed successfully!")
        else:
            logger.error("Processing failed!")
            sys.exit(1)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total processing time: {elapsed_time:.2f}s")

if __name__ == '__main__':
    main()
