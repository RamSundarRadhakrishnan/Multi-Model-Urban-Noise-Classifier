"""
Integrated Inference Pipeline
Combines pre-processing, audio inference, video inference, and event classification
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import numpy as np
import pandas as pd
import time

from cnn_bilstm_audio_inference import CNNBiLSTMBatchAudioInference

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "pre_inference"))
sys.path.append(str(Path(__file__).parent.parent / "image-classifier" / "video_inference"))

# Import pre-processing modules
from integrated_processor import IntegratedMediaProcessor

# Import audio inference modules

from cnn_bilstm_audio_inference import CNNBiLSTMBatchAudioInference

# Import video inference modules
from video_frame_extractor import VideoFrameExtractor
from dual_yolo_inference import DualYOLOInference
from output_fusion import create_fusion_engine

from late_fusion import fuse_chunk

# Import event classification modules
from event_classifier import EventClassifier, NoiseZone
from event_storage import EventStorage


class IntegratedInferencePipeline:
    """
    Complete pipeline: Pre-processing -> Audio/Video Inference -> Event Classification
    """
    
    def __init__(
        self,
        # Pre-processing config
        input_dir: str,
        preprocessing_output_dir: str,
        chunk_duration: float = 10.0,
        
        # Audio inference config
        audio_checkpoint: str = "../cnn-bilstm-audio/best_cnn_bilstm.pt",
        audio_sample_rate: int = 16000,
        audio_batch_size: int = 8,
        
        # Video inference config
        yolo_standard_model: str = None,
        yolo_mocs_model: str = None,
        yolo_conf_threshold: float = 0.5,
        frame_skip: int = 30,
        aggregation_method: str = 'weighted_average',
        min_detection_percentage: float = 10.0,
        construction_threshold: float = 0.7,
        device: str = 'cuda',
        
        # Event classification config
        default_zone: NoiseZone = NoiseZone.RESIDENTIAL,
        event_output_dir: str = "./output/events",
        
        # Processing flags
        skip_preprocessing: bool = False,
        skip_audio_inference: bool = False,
        skip_video_inference: bool = False
    ):
        """
        Initialize integrated pipeline
        
        Args:
            Pre-processing:
                input_dir: Directory with input media files
                preprocessing_output_dir: Where to save preprocessed chunks
                chunk_duration: Chunk duration in seconds
            
            Audio inference:
                audio_pretrained_model: Path to pretrained SpeechBrain model
                audio_finetuned_weights: Path to fine-tuned weights
                audio_num_classes: Number of audio classes
                audio_class_names: List of class names
                audio_sample_rate: Audio sample rate
                audio_max_length: Max audio length in samples
                audio_batch_size: Batch size for inference
            
            Video inference:
                yolo_standard_model: Path to standard YOLO model
                yolo_mocs_model: Path to MOCS YOLO model
                yolo_conf_threshold: Confidence threshold
                frame_skip: Process every Nth frame
                aggregation_method: Method to aggregate frame results
                min_detection_percentage: Min % of frames for detection
                construction_threshold: Min % for construction site detection
                device: 'cuda' or 'cpu'
            
            Event classification:
                default_zone: Default noise zone
                event_output_dir: Where to save event data
            
            Processing flags:
                skip_preprocessing: Skip preprocessing step
                skip_audio_inference: Skip audio inference
                skip_video_inference: Skip video inference
        """
        self.input_dir = Path(input_dir)
        self.preprocessing_output_dir = Path(preprocessing_output_dir)
        self.event_output_dir = Path(event_output_dir)
        
        self.skip_preprocessing = skip_preprocessing
        self.skip_audio_inference = skip_audio_inference
        self.skip_video_inference = skip_video_inference
        
        # Initialize pre-processor
        if not skip_preprocessing:
            self.preprocessor = IntegratedMediaProcessor(
                input_dir=str(input_dir),
                output_base_dir=str(preprocessing_output_dir),
                chunk_duration=chunk_duration
            )
        else:
            self.preprocessor = None
        
        # Initialize audio inference
        if not skip_audio_inference:
            if audio_checkpoint is None:
                raise ValueError("audio_checkpoint is required when audio inference is enabled")

            self.audio_processor = CNNBiLSTMBatchAudioInference(
                checkpoint_path=audio_checkpoint,
                sample_rate=audio_sample_rate,
                batch_size=audio_batch_size,
                device=device
            )

        else:
            self.audio_processor = None
        
        # Initialize video inference
        if not skip_video_inference and yolo_standard_model:
            self.frame_extractor = VideoFrameExtractor(frame_skip=frame_skip)
            self.video_inference = DualYOLOInference(
                standard_model_path=yolo_standard_model,
                mocs_model_path=yolo_mocs_model,
                conf_threshold=yolo_conf_threshold,
                device=device
            )
            self.fusion_engine = create_fusion_engine(
                aggregation_method=aggregation_method,
                min_detection_percentage=min_detection_percentage,
                construction_threshold=construction_threshold
            )
        else:
            self.frame_extractor = None
            self.video_inference = None
            self.fusion_engine = None
        
        # Initialize event classifier and storage
        self.event_classifier = EventClassifier(zone=default_zone)
        self.event_storage = EventStorage(storage_dir=str(event_output_dir))
        
        # Store configuration
        self.config = {
            'chunk_duration': chunk_duration,
            'audio_config': {
                'backend': 'cnn_bilstm',
                'checkpoint': audio_checkpoint,
                'sample_rate': audio_sample_rate,
                'batch_size': audio_batch_size
            } if not skip_audio_inference else None,
            'video_config': {
                'conf_threshold': yolo_conf_threshold,
                'frame_skip': frame_skip,
                'aggregation_method': aggregation_method,
                'construction_threshold': construction_threshold
            } if not skip_video_inference else None,
            'default_zone': default_zone.value
        }
        
        # Performance tracking
        self.performance_metrics = {
            'preprocessing': {'duration': 0, 'files_processed': 0},
            'audio_inference': {'duration': 0, 'chunks_processed': 0},
            'video_inference': {'duration': 0, 'chunks_processed': 0},
            'data_matching': {'duration': 0, 'chunks_matched': 0},
            'event_classification': {'duration': 0, 'events_classified': 0},
            'total_pipeline': {'duration': 0}
        }
    
    def _default_audio_classes(self) -> List[str]:
        """Default audio class names"""
        return [
            "crowd-noise", "generator", "motorvehicle-horn", "mobile-music",
            "community-radio", "construction-site", "motorvehicle-siren", "car-alarm"
        ]
    
    def run_preprocessing(self) -> Dict:
        """
        Run preprocessing step
        
        Returns:
            Dictionary with preprocessing results
        """
        start_time = time.time()
        
        if self.skip_preprocessing or self.preprocessor is None:
            print("⏭️  Skipping preprocessing (using existing chunks)")
            return {}
        
        print("🔄 Starting preprocessing...")
        results = self.preprocessor.process_all(audio_format='wav')
        
        elapsed = time.time() - start_time
        self.performance_metrics['preprocessing']['duration'] = elapsed
        self.performance_metrics['preprocessing']['files_processed'] = len(results)
        
        print(f"✅ Preprocessing complete: {len(results)} files processed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        return results
    
    def run_audio_inference(self, audio_chunks_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Run audio inference on audio chunks
        
        Args:
            audio_chunks_dir: Directory with audio chunks (uses default if None)
            
        Returns:
            DataFrame with audio inference results
        """
        start_time = time.time()
        
        if self.skip_audio_inference or self.audio_processor is None:
            print("⏭️  Skipping audio inference")
            return pd.DataFrame()
        
        if audio_chunks_dir is None:
            audio_chunks_dir = self.preprocessing_output_dir / "audio_chunks"
        
        print(f"🎵 Starting audio inference on: {audio_chunks_dir}")
        
        # Create output directory
        audio_output_dir = self.event_output_dir / "audio_inference"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run inference
        results_df = self.audio_processor.process_directory(
            input_dir=str(audio_chunks_dir),
            output_dir=str(audio_output_dir),
            save_format='both'
        )
        
        # Remove duplicates based on filename (keep first occurrence)
        initial_count = len(results_df)
        results_df = results_df.drop_duplicates(subset=['filename'], keep='first')
        duplicates_removed = initial_count - len(results_df)
        
        if duplicates_removed > 0:
            print(f"  ⚠️  Removed {duplicates_removed} duplicate audio entries")
        
        elapsed = time.time() - start_time
        self.performance_metrics['audio_inference']['duration'] = elapsed
        self.performance_metrics['audio_inference']['chunks_processed'] = len(results_df)
        
        print(f"✅ Audio inference complete: {len(results_df)} chunks processed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        return results_df
    
    def run_video_inference(self, video_chunks_dir: Optional[str] = None) -> List:
        """
        Run video inference on video chunks
        
        Args:
            video_chunks_dir: Directory with video chunks (uses default if None)
            
        Returns:
            List of video inference results
        """
        start_time = time.time()
        
        if self.skip_video_inference or self.video_inference is None:
            print("⏭️  Skipping video inference")
            return []
        
        if video_chunks_dir is None:
            video_chunks_dir = self.preprocessing_output_dir / "video_chunks"
        
        print(f"🎬 Starting video inference on: {video_chunks_dir}")
        
        # Get all video files
        video_files = list(Path(video_chunks_dir).glob("*.mp4"))
        print(f"Found {len(video_files)} video chunks")
        
        results = []
        for i, video_path in enumerate(video_files, 1):
            if i == 1 or i % 10 == 0 or i == len(video_files):  # Show progress every 10 files
                print(f"  Processing {i}/{len(video_files)}: {video_path.name}")
            
            # Extract frames
            frames = list(self.frame_extractor.extract_frames(str(video_path)))
            
            # Run inference
            frame_results = self.video_inference.infer_frames(frames)
            
            # Fuse results
            video_info = self.frame_extractor.get_video_info(str(video_path))
            video_result = self.fusion_engine.fuse_video_results(
                video_path=str(video_path),
                frame_results=frame_results,
                video_metadata=video_info
            )
            
            results.append(video_result)
        
        elapsed = time.time() - start_time
        self.performance_metrics['video_inference']['duration'] = elapsed
        self.performance_metrics['video_inference']['chunks_processed'] = len(results)
        
        print(f"✅ Video inference complete: {len(results)} chunks processed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        return results
    
    def match_audio_video_chunks(
        self,
        audio_results: pd.DataFrame,
        video_results: List,
        noise_summaries: Dict
    ) -> List[Dict]:
        """
        Match audio and video results for the same time chunks
        
        Args:
            audio_results: Audio inference DataFrame
            video_results: List of video inference results
            noise_summaries: Noise analysis summaries from preprocessing
            
        Returns:
            List of matched chunk data
        """
        start_time = time.time()
        print("🔗 Matching audio and video chunks...")
        
        matched_chunks = []
        filtered_count = 0
        
        # Create lookup for video results by filename
        video_lookup = {}
        for vr in video_results:
            video_filename = Path(vr.video_path).name
            video_lookup[video_filename] = vr
        
        # Match each audio result
        for _, audio_row in audio_results.iterrows():
            audio_filename = audio_row['filename']
            
            # Parse chunk info from filename (e.g., "file_chunk_0000.wav")
            base_name = audio_filename.replace('.wav', '').replace('_chunk_', '_chunk_')
            
            # Find corresponding video chunk
            video_filename = base_name + '.mp4'
            video_result = video_lookup.get(video_filename)
            
            # Find noise analysis for this chunk
            # Extract original file and chunk index
            parts = base_name.split('_chunk_')
            if len(parts) == 2:
                original_file = parts[0] + '.mp4'
                chunk_idx = int(parts[1])
                
                noise_data = None
                if original_file in noise_summaries:
                    summary = noise_summaries[original_file]
                    if 'analysis' in summary and chunk_idx < len(summary['analysis']):
                        noise_data = summary['analysis'][chunk_idx]
            else:
                noise_data = None
                chunk_idx = 0
            
            # Compile matched data
            matched_chunk = {
                'chunk_id': base_name,
                'chunk_index': chunk_idx,
                'audio_filename': audio_filename,
                'video_filename': video_filename if video_result else None,
                
                # Audio inference
                'audio_predicted_class': audio_row['predicted_class_name'],
                'audio_confidence': audio_row['confidence'],
                'audio_probabilities': audio_row['probability_vector'] if 'probability_vector' in audio_row else None,
                
                # Video inference
                'video_result': video_result,
                'is_construction_site': video_result.is_construction_site if video_result else False,
                'construction_confidence': video_result.construction_confidence if video_result else 0.0,
                
                # Noise analysis
                'noise_level_db': noise_data.get('a_weighted_spl', 0) if noise_data else 0,
                'leq_db': noise_data.get('leq_a', 0) if noise_data else 0,
                'noise_data': noise_data
            }
            
            # Only add chunks with valid noise levels (filter out -inf, inf, nan)
            noise_level = matched_chunk['noise_level_db']
            if np.isfinite(noise_level) and noise_level >= 0:
                matched_chunks.append(matched_chunk)
            else:
                filtered_count += 1
                if filtered_count <= 5:  # Only show first 5 warnings
                    print(f"  ⚠️  Skipping {matched_chunk['chunk_id']}: invalid noise level ({noise_level})")
        
        elapsed = time.time() - start_time
        self.performance_metrics['data_matching']['duration'] = elapsed
        self.performance_metrics['data_matching']['chunks_matched'] = len(matched_chunks)
        
        # Remove any duplicate chunk_ids (shouldn't happen, but safety check)
        chunk_ids_seen = set()
        deduplicated_chunks = []
        duplicates_in_matching = 0
        
        for chunk in matched_chunks:
            if chunk['chunk_id'] not in chunk_ids_seen:
                chunk_ids_seen.add(chunk['chunk_id'])
                deduplicated_chunks.append(chunk)
            else:
                duplicates_in_matching += 1
        
        if duplicates_in_matching > 0:
            print(f"  ⚠️  Removed {duplicates_in_matching} duplicate chunks during matching")
            matched_chunks = deduplicated_chunks
            self.performance_metrics['data_matching']['chunks_matched'] = len(matched_chunks)
        
        print(f"✅ Matched {len(matched_chunks)} chunks in {elapsed:.2f}s")
        if filtered_count > 0:
            print(f"   ⚠️  Filtered out {filtered_count} chunks with invalid noise levels")
            if filtered_count > 5:
                print(f"   (showing first 5 warnings only)")
        return matched_chunks
    
    def classify_events(
        self,
        matched_chunks: List[Dict],
        zone: Optional[NoiseZone] = None,
        base_timestamp: Optional[datetime] = None
    ) -> List:
        """
        Classify events from matched chunk data
        
        Args:
            matched_chunks: List of matched audio/video/noise data
            zone: Noise zone (uses default if None)
            base_timestamp: Base timestamp for events (uses current time if None)
            
        Returns:
            List of EventClassification objects
        """
        start_time = time.time()
        print("🏷️  Classifying events...")
        
        if base_timestamp is None:
            base_timestamp = datetime.now()
        
        if zone is None:
            zone = self.event_classifier.zone
        
        classified_events = []
        
        for chunk in matched_chunks:
            # Calculate timestamp for this chunk
            chunk_offset_seconds = chunk['chunk_index'] * self.config['chunk_duration']
            chunk_timestamp = base_timestamp.replace(
                second=int(base_timestamp.second + chunk_offset_seconds) % 60,
                minute=int(base_timestamp.minute + chunk_offset_seconds // 60) % 60
            )
            
            # Prepare audio classes dictionary
            fusion = fuse_chunk(chunk, alpha=0.6)

            audio_classes = {
                fusion["audio_class"]: fusion["fusion_score"]
            }
            
            # Prepare video classes dictionary
            video_classes = {}
            if chunk['video_result']:
                for detection in chunk['video_result'].standard_detections[:5]:
                    video_classes[detection.class_name] = detection.confidence
            
            # Classify event
            event = self.event_classifier.classify_event(
                event_id=chunk['chunk_id'],
                timestamp=chunk_timestamp,
                noise_level_db=chunk['noise_level_db'],
                audio_classes=audio_classes,
                video_classes=video_classes,
                is_construction_site=chunk['is_construction_site'],
                construction_confidence=chunk['construction_confidence'],
                zone=zone
            )


            event.fusion_metadata = fusion
            
            classified_events.append(event)
        
        # Remove duplicate events based on event_id (final safety check)
        event_ids_seen = set()
        deduplicated_events = []
        duplicates_in_classification = 0
        
        for event in classified_events:
            if event.event_id not in event_ids_seen:
                event_ids_seen.add(event.event_id)
                deduplicated_events.append(event)
            else:
                duplicates_in_classification += 1
        
        if duplicates_in_classification > 0:
            print(f"  ⚠️  Removed {duplicates_in_classification} duplicate events")
            classified_events = deduplicated_events
        
        elapsed = time.time() - start_time
        self.performance_metrics['event_classification']['duration'] = elapsed
        self.performance_metrics['event_classification']['events_classified'] = len(classified_events)
        
        print(f"✅ Classified {len(classified_events)} events in {elapsed:.2f}s")
        return classified_events
    
    def run_full_pipeline(
        self,
        zone: Optional[NoiseZone] = None,
        base_timestamp: Optional[datetime] = None,
        save_results: bool = True
    ) -> Tuple[List, EventStorage]:
        """
        Run complete integrated pipeline
        
        Args:
            zone: Noise zone for classification
            base_timestamp: Base timestamp for events
            save_results: Whether to save results to files
            
        Returns:
            (classified_events, event_storage)
        """
        pipeline_start = time.time()
        
        print("=" * 80)
        print("🚀 STARTING INTEGRATED INFERENCE PIPELINE")
        print("=" * 80)
        
        # Step 1: Preprocessing
        preprocessing_results = self.run_preprocessing()
        
        # Step 2: Audio Inference
        audio_results = self.run_audio_inference()
        
        # Step 3: Video Inference
        video_results = self.run_video_inference()
        
        # Step 4: Load noise summaries
        print("📊 Loading noise analysis summaries...")
        noise_summaries = {}
        if not self.skip_preprocessing and preprocessing_results:
            for filename in preprocessing_results.keys():
                noise_summary = self.preprocessor.get_noise_summary(filename)
                if 'error' not in noise_summary:
                    noise_summaries[filename] = noise_summary
        
        # Step 5: Match chunks
        matched_chunks = self.match_audio_video_chunks(
            audio_results, video_results, noise_summaries
        )
        
        # Step 6: Classify events
        classified_events = self.classify_events(
            matched_chunks, zone, base_timestamp
        )
        
        # Step 7: Store events
        print("💾 Storing classified events...")
        self.event_storage.add_events(classified_events)
        
        # Step 8: Save results
        if save_results:
            print("📁 Saving results...")
            json_path = self.event_storage.save_json()
            csv_path = self.event_storage.save_csv()
            detailed_csv_path = self.event_storage.save_detailed_csv()
            violations_path = self.event_storage.save_violations_only()
            summary_path = self.event_storage.generate_summary_report()
            
            print(f"  ✓ Events JSON: {json_path}")
            print(f"  ✓ Events CSV: {csv_path}")
            print(f"  ✓ Detailed CSV: {detailed_csv_path}")
            print(f"  ✓ Violations JSON: {violations_path}")
            print(f"  ✓ Summary Report: {summary_path}")
        
        # Calculate total time
        total_elapsed = time.time() - pipeline_start
        self.performance_metrics['total_pipeline']['duration'] = total_elapsed
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Total events: {len(classified_events)}")
        print(f"Compliant: {sum(1 for e in classified_events if e.is_compliant)}")
        print(f"Violations: {sum(1 for e in classified_events if not e.is_compliant)}")
        print(f"\n⏱️  Total Time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
        print("=" * 80)
        
        return classified_events, self.event_storage
    
    def get_performance_report(self) -> Dict:
        """
        Get detailed performance metrics
        
        Returns:
            Dictionary with performance statistics
        """
        metrics = self.performance_metrics.copy()
        
        # Calculate per-item times
        if metrics['audio_inference']['chunks_processed'] > 0:
            metrics['audio_inference']['time_per_chunk'] = (
                metrics['audio_inference']['duration'] / 
                metrics['audio_inference']['chunks_processed']
            )
        
        if metrics['video_inference']['chunks_processed'] > 0:
            metrics['video_inference']['time_per_chunk'] = (
                metrics['video_inference']['duration'] / 
                metrics['video_inference']['chunks_processed']
            )
        
        if metrics['event_classification']['events_classified'] > 0:
            metrics['event_classification']['time_per_event'] = (
                metrics['event_classification']['duration'] / 
                metrics['event_classification']['events_classified']
            )
        
        # Calculate percentages
        total_time = metrics['total_pipeline']['duration']
        if total_time > 0:
            metrics['preprocessing']['percentage'] = (
                metrics['preprocessing']['duration'] / total_time * 100
            )
            metrics['audio_inference']['percentage'] = (
                metrics['audio_inference']['duration'] / total_time * 100
            )
            metrics['video_inference']['percentage'] = (
                metrics['video_inference']['duration'] / total_time * 100
            )
            metrics['data_matching']['percentage'] = (
                metrics['data_matching']['duration'] / total_time * 100
            )
            metrics['event_classification']['percentage'] = (
                metrics['event_classification']['duration'] / total_time * 100
            )
        
        return metrics
    
    def print_performance_report(self):
        """Print formatted performance report"""
        metrics = self.get_performance_report()
        
        print("\n" + "=" * 80)
        print("⏱️  PERFORMANCE REPORT")
        print("=" * 80)
        
        total_time = metrics['total_pipeline']['duration']
        
        print(f"\n{'Stage':<25} {'Time (s)':<12} {'Time (min)':<12} {'% Total':<10} {'Items':<10} {'s/item':<10}")
        print("-" * 80)
        
        # Preprocessing
        if metrics['preprocessing']['duration'] > 0:
            print(f"{'Pre-processing':<25} "
                  f"{metrics['preprocessing']['duration']:<12.2f} "
                  f"{metrics['preprocessing']['duration']/60:<12.2f} "
                  f"{metrics['preprocessing'].get('percentage', 0):<10.1f} "
                  f"{metrics['preprocessing']['files_processed']:<10} "
                  f"{'-':<10}")
        
        # Audio Inference
        if metrics['audio_inference']['duration'] > 0:
            print(f"{'Audio Inference':<25} "
                  f"{metrics['audio_inference']['duration']:<12.2f} "
                  f"{metrics['audio_inference']['duration']/60:<12.2f} "
                  f"{metrics['audio_inference'].get('percentage', 0):<10.1f} "
                  f"{metrics['audio_inference']['chunks_processed']:<10} "
                  f"{metrics['audio_inference'].get('time_per_chunk', 0):<10.3f}")
        
        # Video Inference
        if metrics['video_inference']['duration'] > 0:
            print(f"{'Video Inference':<25} "
                  f"{metrics['video_inference']['duration']:<12.2f} "
                  f"{metrics['video_inference']['duration']/60:<12.2f} "
                  f"{metrics['video_inference'].get('percentage', 0):<10.1f} "
                  f"{metrics['video_inference']['chunks_processed']:<10} "
                  f"{metrics['video_inference'].get('time_per_chunk', 0):<10.3f}")
        
        # Data Matching
        if metrics['data_matching']['duration'] > 0:
            print(f"{'Data Matching':<25} "
                  f"{metrics['data_matching']['duration']:<12.2f} "
                  f"{metrics['data_matching']['duration']/60:<12.2f} "
                  f"{metrics['data_matching'].get('percentage', 0):<10.1f} "
                  f"{metrics['data_matching']['chunks_matched']:<10} "
                  f"{'-':<10}")
        
        # Event Classification
        if metrics['event_classification']['duration'] > 0:
            print(f"{'Event Classification':<25} "
                  f"{metrics['event_classification']['duration']:<12.2f} "
                  f"{metrics['event_classification']['duration']/60:<12.2f} "
                  f"{metrics['event_classification'].get('percentage', 0):<10.1f} "
                  f"{metrics['event_classification']['events_classified']:<10} "
                  f"{metrics['event_classification'].get('time_per_event', 0):<10.4f}")
        
        print("-" * 80)
        print(f"{'TOTAL PIPELINE':<25} "
              f"{total_time:<12.2f} "
              f"{total_time/60:<12.2f} "
              f"{'100.0':<10} "
              f"{'-':<10} "
              f"{'-':<10}")
        
        print("=" * 80)
