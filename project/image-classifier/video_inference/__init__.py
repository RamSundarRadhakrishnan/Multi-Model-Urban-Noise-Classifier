"""
Dual YOLO Video Inference System

A comprehensive system for processing videos using both standard YOLOv8 and 
fine-tuned MOCS models to identify objects and construction sites.

Main components:
- VideoFrameExtractor: Extract frames from videos
- DualYOLOInference: Run inference with both models
- OutputFusion: Fuse and aggregate results
- BatchVideoProcessor: Process multiple videos

Quick example:
    >>> from video_inference import process_video_directory
    >>> results = process_video_directory(
    ...     directory_path='./videos',
    ...     standard_model_path='yolov8n.pt',
    ...     mocs_model_path='mocs_best.pt',
    ...     output_dir='./results'
    ... )
"""

from .video_frame_extractor import VideoFrameExtractor, batch_extract_frames
from .dual_yolo_inference import (
    DualYOLOInference, 
    DetectionResult, 
    FrameInferenceResult,
    CONSTRUCTION_EQUIPMENT
)
from .output_fusion import (
    OutputFusion,
    AggregatedDetection,
    VideoInferenceResult,
    AggregationMethod,
    create_fusion_engine
)
from .batch_processor import (
    BatchVideoProcessor,
    process_video_directory
)

__version__ = '1.0.0'
__author__ = 'Urban Noise Classifier Team'
__all__ = [
    # Frame extraction
    'VideoFrameExtractor',
    'batch_extract_frames',
    
    # Inference
    'DualYOLOInference',
    'DetectionResult',
    'FrameInferenceResult',
    'CONSTRUCTION_EQUIPMENT',
    
    # Fusion
    'OutputFusion',
    'AggregatedDetection',
    'VideoInferenceResult',
    'AggregationMethod',
    'create_fusion_engine',
    
    # Batch processing
    'BatchVideoProcessor',
    'process_video_directory',
]
