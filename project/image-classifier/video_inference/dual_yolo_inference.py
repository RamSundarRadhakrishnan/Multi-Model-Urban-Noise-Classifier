"""
Dual YOLO Inference Module
Performs inference using both standard YOLOv8 and fine-tuned MOCS model.
Includes construction site detection logic.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import logging
from dataclasses import dataclass, field
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MOCS construction equipment classes (excluding workers and other vehicles)
CONSTRUCTION_EQUIPMENT = {
    'Static crane', 'Hanging head', 'Crane', 'Roller', 'Bulldozer', 
    'Excavator', 'Truck', 'Loader', 'Pump truck', 'Concrete mixer', 
    'Pile driving'
}


@dataclass
class DetectionResult:
    """Detection result for a single frame."""
    frame_number: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    model_source: str  # 'standard' or 'mocs'
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'frame_number': self.frame_number,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'model_source': self.model_source
        }


@dataclass
class FrameInferenceResult:
    """Inference result for a single frame from both models."""
    frame_number: int
    standard_detections: List[DetectionResult] = field(default_factory=list)
    mocs_detections: List[DetectionResult] = field(default_factory=list)
    is_construction_site: bool = False
    construction_equipment_detected: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'frame_number': self.frame_number,
            'standard_detections': [d.to_dict() for d in self.standard_detections],
            'mocs_detections': [d.to_dict() for d in self.mocs_detections],
            'is_construction_site': self.is_construction_site,
            'construction_equipment_detected': list(self.construction_equipment_detected)
        }


class DualYOLOInference:
    """Perform inference using both standard and MOCS YOLO models."""
    
    def __init__(self, 
                 standard_model_path: str = 'yolov8n.pt',
                 mocs_model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'cuda'):
        """
        Initialize the dual YOLO inference engine.
        
        Args:
            standard_model_path: Path to standard YOLOv8 model
            mocs_model_path: Path to fine-tuned MOCS model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load standard YOLOv8 model
        logger.info(f"Loading standard YOLO model from {standard_model_path}")
        self.standard_model = YOLO(standard_model_path)
        
        # Load MOCS model if provided
        self.mocs_model = None
        if mocs_model_path:
            logger.info(f"Loading MOCS model from {mocs_model_path}")
            self.mocs_model = YOLO(mocs_model_path)
        else:
            logger.warning("No MOCS model path provided. Construction site detection will be disabled.")
    
    def infer_frame(self, frame: np.ndarray, frame_number: int = 0) -> FrameInferenceResult:
        """
        Perform inference on a single frame using both models.
        
        Args:
            frame: Input frame (numpy array)
            frame_number: Frame number for tracking
            
        Returns:
            FrameInferenceResult containing detections from both models
        """
        result = FrameInferenceResult(frame_number=frame_number)
        
        # Standard YOLOv8 inference
        standard_results = self.standard_model(
            frame, 
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        for det in standard_results:
            if det.boxes is not None:
                for box in det.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.standard_model.names[class_id]
                    confidence = float(box.conf[0].cpu().numpy())
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    
                    detection = DetectionResult(
                        frame_number=frame_number,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        model_source='standard'
                    )
                    result.standard_detections.append(detection)
        
        # MOCS model inference (if available)
        if self.mocs_model:
            mocs_results = self.mocs_model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for det in mocs_results:
                if det.boxes is not None:
                    for box in det.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.mocs_model.names[class_id]
                        confidence = float(box.conf[0].cpu().numpy())
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        detection = DetectionResult(
                            frame_number=frame_number,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            model_source='mocs'
                        )
                        result.mocs_detections.append(detection)
                        
                        # Check if it's construction equipment (exclude workers)
                        if class_name in CONSTRUCTION_EQUIPMENT:
                            result.construction_equipment_detected.add(class_name)
                            result.is_construction_site = True
        
        return result
    
    def infer_frames(self, 
                     frames: List[Tuple[int, np.ndarray]]) -> List[FrameInferenceResult]:
        """
        Perform inference on multiple frames.
        
        Args:
            frames: List of (frame_number, frame_array) tuples
            
        Returns:
            List of FrameInferenceResult
        """
        results = []
        
        for frame_num, frame in frames:
            logger.info(f"Processing frame {frame_num}")
            result = self.infer_frame(frame, frame_num)
            results.append(result)
        
        return results
    
    def get_construction_site_summary(self, 
                                     results: List[FrameInferenceResult]) -> dict:
        """
        Get summary of construction site detection across all frames.
        
        Args:
            results: List of frame inference results
            
        Returns:
            Dictionary with construction site summary
        """
        total_frames = len(results)
        construction_frames = sum(1 for r in results if r.is_construction_site)
        
        # Aggregate all equipment detected
        all_equipment = set()
        equipment_counts = {}
        
        for result in results:
            all_equipment.update(result.construction_equipment_detected)
            for equipment in result.construction_equipment_detected:
                equipment_counts[equipment] = equipment_counts.get(equipment, 0) + 1
        
        return {
            'total_frames': total_frames,
            'construction_frames': construction_frames,
            'construction_percentage': (construction_frames / total_frames * 100) if total_frames > 0 else 0,
            'is_construction_site': construction_frames > 0,
            'equipment_detected': list(all_equipment),
            'equipment_frame_counts': equipment_counts
        }
    
    def get_standard_detections_summary(self, 
                                       results: List[FrameInferenceResult]) -> dict:
        """
        Get summary of standard YOLO detections across all frames.
        
        Args:
            results: List of frame inference results
            
        Returns:
            Dictionary with standard detections summary
        """
        class_counts = {}
        class_confidences = {}
        
        for result in results:
            for detection in result.standard_detections:
                class_name = detection.class_name
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    class_confidences[class_name] = []
                
                class_counts[class_name] += 1
                class_confidences[class_name].append(detection.confidence)
        
        # Calculate average confidence per class
        class_avg_confidence = {
            cls: np.mean(confs) for cls, confs in class_confidences.items()
        }
        
        return {
            'total_detections': sum(class_counts.values()),
            'unique_classes': len(class_counts),
            'class_counts': class_counts,
            'class_avg_confidence': class_avg_confidence
        }
