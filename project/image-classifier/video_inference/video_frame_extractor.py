"""
Video Frame Extractor Module
Handles video loading, frame extraction, and preprocessing for YOLO inference.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """Extract frames from video clips for inference."""
    
    def __init__(self, frame_skip: int = 1, resize_dim: Optional[Tuple[int, int]] = None):
        """
        Initialize the frame extractor.
        
        Args:
            frame_skip: Extract every nth frame (1 = all frames, 2 = every other frame)
            resize_dim: Optional resize dimension (width, height)
        """
        self.frame_skip = frame_skip
        self.resize_dim = resize_dim
        
    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        try:
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames based on frame_skip parameter
                if frame_count % self.frame_skip == 0:
                    # Resize if specified
                    if self.resize_dim:
                        frame = cv2.resize(frame, self.resize_dim)
                    
                    yield frame_count, frame
                    extracted_count += 1
                
                frame_count += 1
            
            logger.info(f"Extracted {extracted_count} frames from {frame_count} total frames")
            
        finally:
            cap.release()
    
    def extract_frames_list(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
        """
        Extract all frames as a list (use for smaller videos).
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of (frame_number, frame_array) tuples
        """
        return list(self.extract_frames(video_path))
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        try:
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            logger.info(f"Video info for {video_path.name}: {info}")
            return info
            
        finally:
            cap.release()
    
    def save_frame(self, frame: np.ndarray, output_path: str):
        """
        Save a frame to disk.
        
        Args:
            frame: Frame array
            output_path: Output file path
        """
        cv2.imwrite(output_path, frame)
        logger.info(f"Frame saved to {output_path}")


def batch_extract_frames(video_paths: List[str], 
                         frame_skip: int = 1,
                         resize_dim: Optional[Tuple[int, int]] = None) -> dict:
    """
    Extract frames from multiple videos.
    
    Args:
        video_paths: List of video file paths
        frame_skip: Extract every nth frame
        resize_dim: Optional resize dimension
        
    Returns:
        Dictionary mapping video paths to frame lists
    """
    extractor = VideoFrameExtractor(frame_skip=frame_skip, resize_dim=resize_dim)
    results = {}
    
    for video_path in video_paths:
        logger.info(f"Extracting frames from {video_path}")
        frames = extractor.extract_frames_list(video_path)
        results[video_path] = frames
    
    return results
