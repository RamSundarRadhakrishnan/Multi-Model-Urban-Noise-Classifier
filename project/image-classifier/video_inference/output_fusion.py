"""
Output Fusion and Aggregation Module
Fuses outputs from both YOLO models and aggregates results across video frames.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

from dual_yolo_inference import FrameInferenceResult, DetectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Available aggregation methods for video-level results."""
    MAX_CONFIDENCE = "max_confidence"  # Take detection with highest confidence
    AVERAGE_CONFIDENCE = "average_confidence"  # Average confidence across frames
    MAJORITY_VOTE = "majority_vote"  # Most frequently detected class
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by confidence and frequency
    THRESHOLD_PERCENTAGE = "threshold_percentage"  # Detected in X% of frames


@dataclass
class AggregatedDetection:
    """Aggregated detection result for entire video."""
    class_name: str
    confidence: float
    detection_count: int  # Number of frames where detected
    total_frames: int
    detection_percentage: float
    model_source: str
    avg_bbox: Optional[List[float]] = None  # Average bbox across detections
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'detection_count': self.detection_count,
            'total_frames': self.total_frames,
            'detection_percentage': self.detection_percentage,
            'model_source': self.model_source,
            'avg_bbox': self.avg_bbox
        }


@dataclass
class VideoInferenceResult:
    """Complete inference result for a video."""
    video_path: str
    total_frames: int
    is_construction_site: bool
    construction_confidence: float
    construction_equipment: List[str] = field(default_factory=list)
    standard_detections: List[AggregatedDetection] = field(default_factory=list)
    mocs_detections: List[AggregatedDetection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'is_construction_site': self.is_construction_site,
            'construction_confidence': self.construction_confidence,
            'construction_equipment': self.construction_equipment,
            'standard_detections': [d.to_dict() for d in self.standard_detections],
            'mocs_detections': [d.to_dict() for d in self.mocs_detections],
            'metadata': self.metadata
        }


class OutputFusion:
    """Fuse and aggregate detection results across video frames."""
    
    def __init__(self, 
                 aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                 min_detection_percentage: float = 0.1,
                 construction_threshold: float = 0.3):
        """
        Initialize the output fusion module.
        
        Args:
            aggregation_method: Method to aggregate detections across frames
            min_detection_percentage: Minimum % of frames for a detection to be included
            construction_threshold: Minimum % of frames to classify as construction site
        """
        self.aggregation_method = aggregation_method
        self.min_detection_percentage = min_detection_percentage
        self.construction_threshold = construction_threshold
    
    def aggregate_detections(self, 
                            frame_results: List[FrameInferenceResult],
                            model_source: str = 'standard') -> List[AggregatedDetection]:
        """
        Aggregate detections from multiple frames.
        
        Args:
            frame_results: List of frame inference results
            model_source: Which model's detections to aggregate ('standard' or 'mocs')
            
        Returns:
            List of aggregated detections
        """
        total_frames = len(frame_results)
        
        # Collect all detections by class
        class_detections = {}
        
        for frame_idx, result in enumerate(frame_results):
            detections = (result.standard_detections if model_source == 'standard' 
                         else result.mocs_detections)
            
            # Track which classes appear in this frame
            classes_in_frame = set()
            
            for detection in detections:
                class_name = detection.class_name
                
                if class_name not in class_detections:
                    class_detections[class_name] = {
                        'confidences': [],
                        'bboxes': [],
                        'frames_with_detection': set()
                    }
                
                class_detections[class_name]['confidences'].append(detection.confidence)
                class_detections[class_name]['bboxes'].append(detection.bbox)
                classes_in_frame.add(class_name)
            
            # Mark frames where each class was detected
            for class_name in classes_in_frame:
                class_detections[class_name]['frames_with_detection'].add(frame_idx)
        
        # Aggregate based on selected method
        aggregated = []
        
        for class_name, data in class_detections.items():
            detection_count = len(data['frames_with_detection'])  # Number of frames where detected
            detection_percentage = (detection_count / total_frames) * 100
            
            # Filter out rare detections
            if detection_percentage < self.min_detection_percentage:
                continue
            
            # Calculate aggregated confidence based on method
            confidence = self._calculate_aggregated_confidence(
                data['confidences'],
                detection_count,
                total_frames
            )
            
            # Ensure confidence is clamped to [0, 1]
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Calculate average bbox
            avg_bbox = np.mean(data['bboxes'], axis=0).tolist() if data['bboxes'] else None
            
            aggregated.append(AggregatedDetection(
                class_name=class_name,
                confidence=confidence,
                detection_count=detection_count,
                total_frames=total_frames,
                detection_percentage=detection_percentage,
                model_source=model_source,
                avg_bbox=avg_bbox
            ))
        
        # Sort by confidence (descending)
        aggregated.sort(key=lambda x: x.confidence, reverse=True)
        
        return aggregated
    
    def _calculate_aggregated_confidence(self, 
                                        confidences: List[float],
                                        detection_count: int,
                                        total_frames: int) -> float:
        """
        Calculate aggregated confidence based on aggregation method.
        
        Args:
            confidences: List of confidence scores
            detection_count: Number of frames with detection
            total_frames: Total number of frames
            
        Returns:
            Aggregated confidence score
        """
        if self.aggregation_method == AggregationMethod.MAX_CONFIDENCE:
            return max(confidences)
        
        elif self.aggregation_method == AggregationMethod.AVERAGE_CONFIDENCE:
            return np.mean(confidences)
        
        elif self.aggregation_method == AggregationMethod.MAJORITY_VOTE:
            # Return 1.0 if detected in >50% of frames, else average confidence
            if detection_count > total_frames / 2:
                return 1.0
            else:
                return np.mean(confidences)
        
        elif self.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            # Weight by both confidence and frequency
            avg_confidence = np.mean(confidences)
            frequency_weight = detection_count / total_frames
            return avg_confidence * 0.7 + frequency_weight * 0.3
        
        elif self.aggregation_method == AggregationMethod.THRESHOLD_PERCENTAGE:
            # Binary: 1.0 if above threshold, else average confidence
            detection_percentage = (detection_count / total_frames) * 100
            if detection_percentage >= self.min_detection_percentage:
                return np.mean(confidences)
            else:
                return 0.0
        
        else:
            return np.mean(confidences)
    
    def determine_construction_site(self, 
                                   frame_results: List[FrameInferenceResult]) -> tuple:
        """
        Determine if video shows a construction site based on equipment detections.
        
        Args:
            frame_results: List of frame inference results
            
        Returns:
            Tuple of (is_construction_site, confidence, equipment_list)
        """
        total_frames = len(frame_results)
        construction_frames = sum(1 for r in frame_results if r.is_construction_site)
        
        if total_frames == 0:
            return False, 0.0, []
        
        construction_percentage = (construction_frames / total_frames) * 100
        
        # Collect all equipment detected
        all_equipment = set()
        for result in frame_results:
            all_equipment.update(result.construction_equipment_detected)
        
        # Determine if it's a construction site
        is_construction_site = construction_percentage >= (self.construction_threshold * 100)
        
        # Confidence based on percentage of frames with construction equipment
        confidence = min(construction_percentage / 100, 1.0)
        
        return is_construction_site, confidence, list(all_equipment)
    
    def fuse_video_results(self,
                          video_path: str,
                          frame_results: List[FrameInferenceResult],
                          video_metadata: Optional[dict] = None) -> VideoInferenceResult:
        """
        Fuse all frame results into a single video-level result.
        
        Args:
            video_path: Path to the video file
            frame_results: List of frame inference results
            video_metadata: Optional video metadata
            
        Returns:
            VideoInferenceResult containing aggregated results
        """
        logger.info(f"Fusing results for {video_path} ({len(frame_results)} frames)")
        
        # Aggregate standard YOLO detections
        standard_detections = self.aggregate_detections(frame_results, 'standard')
        
        # Aggregate MOCS detections
        mocs_detections = self.aggregate_detections(frame_results, 'mocs')
        
        # Determine construction site status
        is_construction, construction_conf, equipment = self.determine_construction_site(frame_results)
        
        result = VideoInferenceResult(
            video_path=video_path,
            total_frames=len(frame_results),
            is_construction_site=is_construction,
            construction_confidence=construction_conf,
            construction_equipment=equipment,
            standard_detections=standard_detections,
            mocs_detections=mocs_detections,
            metadata=video_metadata or {}
        )
        
        logger.info(f"Construction site: {is_construction} (confidence: {construction_conf:.2%})")
        logger.info(f"Standard detections: {len(standard_detections)} unique classes")
        logger.info(f"MOCS detections: {len(mocs_detections)} unique classes")
        
        return result
    
    def get_top_detections(self,
                          video_result: VideoInferenceResult,
                          top_n: int = 5,
                          model_source: str = 'standard') -> List[AggregatedDetection]:
        """
        Get top N detections from video result.
        
        Args:
            video_result: Video inference result
            top_n: Number of top detections to return
            model_source: Which model's detections to use
            
        Returns:
            List of top N aggregated detections
        """
        detections = (video_result.standard_detections if model_source == 'standard'
                     else video_result.mocs_detections)
        
        return detections[:top_n]


def create_fusion_engine(aggregation_method: str = "weighted_average",
                        min_detection_percentage: float = 0.1,
                        construction_threshold: float = 0.3) -> OutputFusion:
    """
    Factory function to create an OutputFusion engine.
    
    Args:
        aggregation_method: Name of aggregation method
        min_detection_percentage: Minimum detection percentage threshold
        construction_threshold: Construction site classification threshold
        
    Returns:
        Configured OutputFusion instance
    """
    method_map = {
        "max_confidence": AggregationMethod.MAX_CONFIDENCE,
        "average_confidence": AggregationMethod.AVERAGE_CONFIDENCE,
        "majority_vote": AggregationMethod.MAJORITY_VOTE,
        "weighted_average": AggregationMethod.WEIGHTED_AVERAGE,
        "threshold_percentage": AggregationMethod.THRESHOLD_PERCENTAGE
    }
    
    method = method_map.get(aggregation_method.lower(), AggregationMethod.WEIGHTED_AVERAGE)
    
    return OutputFusion(
        aggregation_method=method,
        min_detection_percentage=min_detection_percentage,
        construction_threshold=construction_threshold
    )
