"""
Batch Video Processor
Process multiple video clips from a directory and save results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
from tqdm import tqdm

from video_frame_extractor import VideoFrameExtractor
from dual_yolo_inference import DualYOLOInference
from output_fusion import OutputFusion, VideoInferenceResult, create_fusion_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchVideoProcessor:
    """Process multiple videos from a directory."""
    
    def __init__(self,
                 standard_model_path: str = 'yolov8n.pt',
                 mocs_model_path: Optional[str] = None,
                 conf_threshold: float = 0.5,
                 frame_skip: int = 30,  # Process every 30th frame (~1 fps for 30fps video)
                 aggregation_method: str = "weighted_average",
                 min_detection_percentage: float = 10.0,
                 construction_threshold: float = 0.3,
                 device: str = 'cuda'):
        """
        Initialize the batch processor.
        
        Args:
            standard_model_path: Path to standard YOLO model
            mocs_model_path: Path to fine-tuned MOCS model
            conf_threshold: Confidence threshold for detections
            frame_skip: Process every nth frame
            aggregation_method: Method to aggregate results
            min_detection_percentage: Minimum detection percentage
            construction_threshold: Construction site threshold
            device: Device for inference
        """
        self.frame_extractor = VideoFrameExtractor(frame_skip=frame_skip)
        
        self.inference_engine = DualYOLOInference(
            standard_model_path=standard_model_path,
            mocs_model_path=mocs_model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        
        self.fusion_engine = create_fusion_engine(
            aggregation_method=aggregation_method,
            min_detection_percentage=min_detection_percentage,
            construction_threshold=construction_threshold
        )
        
        self.frame_skip = frame_skip
        
    def process_video(self, video_path: str) -> VideoInferenceResult:
        """
        Process a single video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoInferenceResult
        """
        logger.info(f"Processing video: {video_path}")
        
        # Get video metadata
        video_info = self.frame_extractor.get_video_info(video_path)
        
        # Extract frames
        frames = list(self.frame_extractor.extract_frames(video_path))
        logger.info(f"Extracted {len(frames)} frames")
        
        # Perform inference on all frames
        frame_results = self.inference_engine.infer_frames(frames)
        
        # Fuse results
        video_result = self.fusion_engine.fuse_video_results(
            video_path=video_path,
            frame_results=frame_results,
            video_metadata=video_info
        )
        
        return video_result
    
    def process_directory(self, 
                         directory_path: str,
                         output_dir: Optional[str] = None,
                         video_extensions: List[str] = None) -> List[VideoInferenceResult]:
        """
        Process all videos in a directory.
        
        Args:
            directory_path: Directory containing video files
            output_dir: Directory to save results (optional)
            video_extensions: List of video file extensions to process
            
        Returns:
            List of VideoInferenceResult
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(directory.glob(f'*{ext}'))
            video_files.extend(directory.glob(f'*{ext.upper()}'))
        
        video_files = sorted(set(video_files))  # Remove duplicates and sort
        
        logger.info(f"Found {len(video_files)} video files in {directory}")
        
        if len(video_files) == 0:
            logger.warning("No video files found!")
            return []
        
        # Process all videos
        results = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                result = self.process_video(str(video_path))
                results.append(result)
                logger.info(f"Successfully processed: {video_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {e}")
                continue
        
        # Save results if output directory specified
        if output_dir:
            self.save_results(results, output_dir)
        
        return results
    
    def save_results(self, 
                    results: List[VideoInferenceResult],
                    output_dir: str,
                    save_json: bool = True,
                    save_csv: bool = True):
        """
        Save processing results to disk.
        
        Args:
            results: List of video inference results
            output_dir: Output directory
            save_json: Save detailed JSON results
            save_csv: Save summary CSV
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        if save_json:
            json_path = output_path / f"video_inference_results_{timestamp}.json"
            
            json_data = {
                'timestamp': timestamp,
                'total_videos': len(results),
                'results': [r.to_dict() for r in results]
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved JSON results to: {json_path}")
        
        # Save summary CSV
        if save_csv:
            csv_path = output_path / f"video_inference_summary_{timestamp}.csv"
            
            summary_data = []
            for result in results:
                # Format all standard detections
                standard_classes = [f"{d.class_name} ({d.confidence:.2%})" 
                                   for d in result.standard_detections]
                
                # Format all MOCS detections
                mocs_classes = [f"{d.class_name} ({d.confidence:.2%})" 
                               for d in result.mocs_detections]
                
                row = {
                    'video_path': Path(result.video_path).name,
                    'total_frames': result.total_frames,
                    'is_construction_site': result.is_construction_site,
                    'construction_confidence': f"{result.construction_confidence:.2%}",
                    'construction_equipment': ', '.join(result.construction_equipment),
                    'standard_detections': '; '.join(standard_classes) if standard_classes else 'None',
                    'mocs_detections': '; '.join(mocs_classes) if mocs_classes else 'None',
                    'num_standard_classes': len(result.standard_detections),
                    'num_mocs_classes': len(result.mocs_detections)
                }
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved CSV summary to: {csv_path}")
        
        # Save detailed per-video reports
        details_dir = output_path / f"detailed_reports_{timestamp}"
        details_dir.mkdir(exist_ok=True)
        
        for result in results:
            video_name = Path(result.video_path).stem
            detail_path = details_dir / f"{video_name}_report.json"
            
            with open(detail_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved detailed reports to: {details_dir}")
    
    def get_summary_statistics(self, results: List[VideoInferenceResult]) -> dict:
        """
        Get summary statistics across all processed videos.
        
        Args:
            results: List of video inference results
            
        Returns:
            Dictionary with summary statistics
        """
        total_videos = len(results)
        construction_videos = sum(1 for r in results if r.is_construction_site)
        
        # Collect all detected classes
        all_standard_classes = set()
        all_construction_equipment = set()
        
        for result in results:
            for detection in result.standard_detections:
                all_standard_classes.add(detection.class_name)
            all_construction_equipment.update(result.construction_equipment)
        
        # Average confidence scores
        avg_construction_conf = (
            sum(r.construction_confidence for r in results) / total_videos 
            if total_videos > 0 else 0
        )
        
        return {
            'total_videos_processed': total_videos,
            'construction_site_videos': construction_videos,
            'construction_percentage': (construction_videos / total_videos * 100) if total_videos > 0 else 0,
            'avg_construction_confidence': avg_construction_conf,
            'unique_standard_classes': len(all_standard_classes),
            'standard_classes_detected': sorted(all_standard_classes),
            'unique_construction_equipment': len(all_construction_equipment),
            'construction_equipment_detected': sorted(all_construction_equipment)
        }


def process_video_directory(directory_path: str,
                           standard_model_path: str = 'yolov8n.pt',
                           mocs_model_path: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           **kwargs) -> List[VideoInferenceResult]:
    """
    Convenience function to process a directory of videos.
    
    Args:
        directory_path: Directory containing videos
        standard_model_path: Path to standard YOLO model
        mocs_model_path: Path to MOCS model
        output_dir: Output directory for results
        **kwargs: Additional arguments for BatchVideoProcessor
        
    Returns:
        List of VideoInferenceResult
    """
    processor = BatchVideoProcessor(
        standard_model_path=standard_model_path,
        mocs_model_path=mocs_model_path,
        **kwargs
    )
    
    results = processor.process_directory(directory_path, output_dir)
    
    # Print summary
    summary = processor.get_summary_statistics(results)
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)
    
    return results
