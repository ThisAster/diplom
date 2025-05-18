# RTSP Video Denoising Streamer

A real-time video streaming application that supports multiple denoising methods including ONNX models, PyTorch Lightning models, and OpenCV bilateral filtering.

## Features

- Real-time video streaming with RTSP
- Multiple denoising methods:
  - ONNX models (CPU/GPU)
  - PyTorch Lightning models (CPU/GPU)
  - OpenCV bilateral filtering
- GPU acceleration support
- Configurable FPS and GPU memory usage
- Split-screen view of original and denoised video

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- FFmpeg installed on your system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd denoise_vlc_rtsp
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg (if not already installed):
- Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Usage

### Basic Usage

```bash
python utils/rtsp_streamer.py --method bilateral --video data/video/walking.mp4
```

### Using ONNX Model

```bash
python utils/rtsp_streamer.py --method onnx --model path/to/model.onnx --video data/video/walking_noised.mp4
```

### Using PyTorch Lightning Model

```bash
python utils/rtsp_streamer.py --method lightning --model models/lightning_model.ckpt --video data/video/walking.mp4
```

### Advanced Options

```bash
python utils/rtsp_streamer.py \
    --method onnx \
    --model models/model.onnx \
    --video data/video/walking.mp4 \
    --fps 60 \
    --gpu-memory-fraction 0.8
```

### Command Line Arguments

- `--method`: Denoising method to use (`bilateral`, `onnx`, or `lightning`)
- `--model`: Path to model file (required for `onnx` and `lightning` methods)
- `--video`: Path to input video file
- `--fps`: Target FPS for streaming (default: 60)
- `--gpu-memory-fraction`: Fraction of GPU memory to use (0.0 to 1.0, default: 0.8)

## Viewing the Stream

1. Using VLC:
```bash
vlc udp://@127.0.0.1:1234
```

2. Using FFplay:
```bash
ffplay udp://127.0.0.1:1234
```

## Performance Tips

1. For best GPU performance:
   - Use `--gpu-memory-fraction 0.8` to leave some memory for system processes
   - Ensure your model is optimized for inference
   - Use appropriate batch sizes for your GPU memory

2. For CPU-only systems:
   - Use the bilateral filter method for better performance
   - Reduce the target FPS if needed

## Troubleshooting

1. If you get CUDA out of memory errors:
   - Reduce the `--gpu-memory-fraction`
   - Use a smaller model
   - Reduce the input video resolution

2. If streaming is slow:
   - Check your network connection
   - Reduce the target FPS
   - Use a more efficient denoising method

3. If the video quality is poor:
   - Adjust the bilateral filter parameters
   - Use a higher quality model
   - Increase the bitrate in FFmpeg settings
