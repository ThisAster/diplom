import cv2
import numpy as np

def add_gaussian_noise(video_path, output_path, noise_level=25):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(frame, noise)
        
        # Write the frame
        out.write(noisy_frame)
    
    # Release everything
    cap.release()
    out.release()

if __name__ == "__main__":
    # Add noise to both videos
    add_gaussian_noise("data/video/traffic_sd.mp4", "data/video/traffic_sd_noised.mp4")
    add_gaussian_noise("data/video/walking.mp4", "data/video/walking_noised.mp4") 