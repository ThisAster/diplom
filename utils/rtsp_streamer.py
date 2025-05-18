import os
import sys
import cv2
import numpy as np
import subprocess
import time
import signal
import threading
from queue import Queue
import onnxruntime as ort
import torch
import argparse
import gc
from models.denoiser import SwinIRLightning
from utils.stream_encryption import StreamEncryption

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

gc.collect()
torch.cuda.empty_cache()

class RTSPStreamer:
    def __init__(self, video_path, target_fps=60, denoise_method='bilateral', model_path=None, gpu_memory_fraction=0.8, encryption_key=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.target_fps = target_fps
        self.frame_queue = Queue(maxsize=120)
        self.running = True
        self.denoise_method = denoise_method
        self.gpu_memory_fraction = gpu_memory_fraction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize encryption if key is provided
        self.encryption = StreamEncryption.from_key_base64(encryption_key) if encryption_key else None
        if self.encryption:
            print("Stream encryption enabled")
            print(f"Encryption key (base64): {self.encryption.get_key_base64()}")
        
        # Initialize model based on method
        if denoise_method == 'onnx' and model_path:
            print("Initializing ONNX Runtime...")
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                print("CUDA is available, using GPU acceleration")
                cuda_options = {
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True
                }
                providers = [
                    ('CUDAExecutionProvider', cuda_options),
                    'CPUExecutionProvider'
                ]
            else:
                print("CUDA not available, using CPU")
                providers = ['CPUExecutionProvider']
            
            self.ort_session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.input_shape = self.ort_session.get_inputs()[0].shape
            print(f"ONNX model loaded. Input shape: {self.input_shape}")
            print(f"Using provider: {self.ort_session.get_providers()[0]}")
            
        elif denoise_method == 'lightning' and model_path:
            print("Initializing PyTorch Lightning model...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                print(f"CUDA memory fraction set to {gpu_memory_fraction}")
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            self.model = SwinIRLightning.load_from_checkpoint(model_path)
            self.model.eval()
            self.model.to(self.device)
            print(f"PyTorch Lightning model loaded on {self.device}")
        
    def process_frame_onnx(self, frame):
        # Preprocess frame for ONNX model
        input_frame = cv2.resize(frame, (640, 480))  # Resize to match model's expected input
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.transpose(input_frame, (2, 0, 1))  # HWC to CHW
        input_frame = np.expand_dims(input_frame, axis=0)  # Add batch dimension
        
        # Run inference with memory optimization
        try:
            outputs = self.ort_session.run(None, {self.input_name: input_frame})
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return frame
        
        # Postprocess output
        denoised = outputs[0][0]  # Remove batch dimension
        denoised = np.transpose(denoised, (1, 2, 0))  # CHW to HWC
        denoised = (denoised * 255.0).astype(np.uint8)
        denoised = cv2.resize(denoised, (self.width, self.height))
        
        return denoised
        
    def process_frame_lightning(self, frame):
        # Preprocess frame for PyTorch model
        input_frame = cv2.resize(frame, (640, 480))  # Resize to match model's expected input
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.transpose(input_frame, (2, 0, 1))  # HWC to CHW
        input_frame = torch.from_numpy(input_frame).unsqueeze(0).to(self.device)
        
        # Run inference with memory optimization
        try:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    output = self.model(input_frame)
        except Exception as e:
            print(f"PyTorch inference error: {e}")
            return frame
        
        # Postprocess output
        denoised = output[0].cpu().numpy()
        denoised = np.transpose(denoised, (1, 2, 0))  # CHW to HWC
        denoised = (denoised * 255.0).astype(np.uint8)
        denoised = cv2.resize(denoised, (self.width, self.height))
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return denoised
        
    def process_frame_bilateral(self, frame):
        # Use bilateral filter for faster denoising with good quality
        denoised = cv2.bilateralFilter(
            frame,
            d=5,  # Diameter of each pixel neighborhood
            sigmaColor=75,  # Filter sigma in the color space
            sigmaSpace=75   # Filter sigma in the coordinate space
        )
        return denoised
        
    def process_frame(self, frame):
        # Choose denoising method
        if self.denoise_method == 'onnx':
            denoised = self.process_frame_onnx(frame)
        elif self.denoise_method == 'lightning':
            denoised = self.process_frame_lightning(frame)
        else:  # bilateral
            denoised = self.process_frame_bilateral(frame)
        
        # Create vertical split screen (left: noisy, right: denoised)
        split_screen = np.hstack((frame, denoised))
        
        # Encrypt if enabled
        if self.encryption:
            split_screen = self.encryption.encrypt_frame(split_screen)
        
        return split_screen
        
    def process_frames_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Process frame
            processed_frame = self.process_frame(frame)
            self.frame_queue.put(processed_frame)
            
            # No sleep to maximize processing speed
            # time.sleep(0.001)
        
    def stream(self):
        try:
            # Create a named pipe for FFmpeg
            pipe_path = '/tmp/video_pipe'
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
            os.mkfifo(pipe_path)
            
            # Start FFmpeg process with optimized settings
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width*2}x{self.height}',
                '-r', str(self.target_fps),
                '-i', pipe_path,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-b:v', '8000k',
                '-maxrate', '10000k',
                '-bufsize', '20000k',
                '-g', '30',
                '-f', 'mpegts',
                '-flush_packets', '1',
                'udp://127.0.0.1:1234?pkt_size=1316'
            ]
            
            print(f"Starting FFmpeg stream at {self.target_fps} FPS...")
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            
            # Start processing thread
            process_thread = threading.Thread(target=self.process_frames_thread)
            process_thread.start()
            
            # Open the pipe for writing
            with open(pipe_path, 'wb') as pipe:
                frame_count = 0
                start_time = time.time()
                last_fps_print = time.time()
                
                while self.running:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get()
                        if self.encryption:
                            # If encrypted, write the encrypted bytes directly
                            pipe.write(frame)
                        else:
                            # If not encrypted, convert to bytes as before
                            pipe.write(frame.tobytes())
                        pipe.flush()
                        
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        
                        # Print FPS every second
                        if time.time() - last_fps_print >= 1.0:
                            current_fps = frame_count / elapsed_time
                            print(f"Current FPS: {current_fps:.2f}")
                            frame_count = 0
                            start_time = time.time()
                            last_fps_print = time.time()
                    
        except KeyboardInterrupt:
            print("\nStopping stream...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False
            if process_thread.is_alive():
                process_thread.join()
            self.cleanup()
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
            
    def cleanup(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cv2.destroyAllWindows()
        gc.collect()

def signal_handler(sig, frame):
    print('Cleaning up...')
    sys.exit(0)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='RTSP Streamer with multiple denoising methods')
    parser.add_argument('--method', type=str, default='bilateral',
                      choices=['bilateral', 'onnx', 'lightning'],
                      help='Denoising method to use')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to model file (required for onnx and lightning methods)')
    parser.add_argument('--fps', type=int, default=60,
                      help='Target FPS for streaming')
    parser.add_argument('--video', type=str, default='data/video/traffic_sd_noised.mp4',
                      help='Path to input video file')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.8,
                      help='Fraction of GPU memory to use (0.0 to 1.0)')
    parser.add_argument('--encryption-key', type=str, default=None,
                      help='Base64 encoded encryption key (if not provided, encryption will be disabled)')
    
    args = parser.parse_args()
    
    # Set up signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    # First add noise to the video if it doesn't exist
    if not os.path.exists(args.video):
        print("Adding noise to video...")
        from add_noise import add_gaussian_noise
        add_gaussian_noise(args.video.replace('_noised', ''), args.video, noise_level=30)
    
    # Create streamer with chosen method
    streamer = RTSPStreamer(
        args.video,
        target_fps=args.fps,
        denoise_method=args.method,
        model_path=args.model,
        gpu_memory_fraction=args.gpu_memory_fraction,
        encryption_key=args.encryption_key
    )
    streamer.stream() 