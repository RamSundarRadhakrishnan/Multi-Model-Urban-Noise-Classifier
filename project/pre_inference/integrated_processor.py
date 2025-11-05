"""
Integrated Media Processor with Chunking and Audio Analysis
Combines separation, chunking, and A-weighted noise level analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import subprocess

from media_processor import MediaProcessor
from media_chunker import MediaChunker
from audio_analyzer import AudioAnalyzer


class IntegratedMediaProcessor:
    """
    Integrated processor that handles:
    1. Audio/video separation
    2. Chunking into configurable segments
    3. A-weighted noise level analysis
    """
    
    def __init__(self, input_dir: str, output_base_dir: str = None, 
                 chunk_duration: float = 10.0):
        """
        Initialize the integrated processor.
        
        Args:
            input_dir: Directory containing input MP4 files
            output_base_dir: Base directory for output files
            chunk_duration: Duration of each chunk in seconds (default: 10.0)
        """
        self.input_dir = Path(input_dir)
        
        if output_base_dir is None:
            self.output_base_dir = self.input_dir.parent / 'processed_media'
        else:
            self.output_base_dir = Path(output_base_dir)
        
        # Initialize components
        self.media_processor = MediaProcessor(input_dir, str(self.output_base_dir))
        self.chunker = MediaChunker(chunk_duration=chunk_duration)
        self.audio_analyzer = AudioAnalyzer()
        
        # Create chunk directories
        self.audio_chunks_dir = self.output_base_dir / 'audio_chunks'
        self.video_chunks_dir = self.output_base_dir / 'video_chunks'
        self.chunk_metadata_dir = self.output_base_dir / 'chunk_metadata'
        self.noise_analysis_dir = self.output_base_dir / 'noise_analysis'
        
        self._create_directories()
        
        # Configuration
        self.chunk_duration = chunk_duration
    
    def _create_directories(self):
        """Create all necessary output directories."""
        for directory in [self.audio_chunks_dir, self.video_chunks_dir, 
                         self.chunk_metadata_dir, self.noise_analysis_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_file_complete(self, video_file: str, audio_format: str = 'wav') -> Dict:
        """
        Complete processing pipeline for a single file:
        1. Separate audio and video
        2. Chunk both streams
        3. Analyze audio chunks for noise levels
        
        Args:
            video_file: Name of the video file
            audio_format: Audio output format
            
        Returns:
            Dictionary with complete processing results
        """
        print(f"\n{'='*70}")
        print(f"Processing: {video_file}")
        print(f"{'='*70}")
        
        # Step 1: Separate audio and video
        print("\n[1/4] Separating audio and video...")
        success, base_metadata = self.media_processor.process_file(video_file, audio_format)
        
        if not success:
            print(f"✗ Failed to separate audio/video for {video_file}")
            return {'success': False, 'error': 'Separation failed'}
        
        # Get paths to separated files
        audio_path, video_path, metadata = self.media_processor.get_aligned_pair(video_file)
        
        if not audio_path.exists() or not video_path.exists():
            print(f"✗ Separated files not found")
            return {'success': False, 'error': 'Separated files not found'}
        
        base_name = Path(video_file).stem
        
        # Step 2: Chunk audio and video
        print(f"\n[2/4] Chunking into {self.chunk_duration}s segments...")
        audio_chunks, video_chunks = self.chunker.chunk_media_pair(
            audio_path, video_path,
            self.audio_chunks_dir, self.video_chunks_dir,
            base_name
        )
        
        if not audio_chunks:
            print(f"✗ No chunks created")
            return {'success': False, 'error': 'Chunking failed'}
        
        # Step 3: Analyze audio chunks
        print(f"\n[3/4] Analyzing audio chunks (A-weighted noise levels)...")
        noise_analysis = self.audio_analyzer.analyze_audio_clips_batch(audio_chunks)
        
        # Step 4: Create comprehensive metadata
        print(f"\n[4/4] Creating metadata...")
        chunk_metadata = self.chunker.create_chunk_metadata(
            audio_chunks, video_chunks, video_file, base_name
        )
        
        # Merge noise analysis into chunk metadata
        for i, (chunk_meta, noise_data) in enumerate(zip(chunk_metadata, noise_analysis)):
            chunk_meta['noise_analysis'] = noise_data
        
        # Save chunk metadata
        chunk_meta_file = self.chunk_metadata_dir / f"{base_name}_chunks.json"
        with open(chunk_meta_file, 'w') as f:
            json.dump({
                'original_file': video_file,
                'base_name': base_name,
                'chunk_duration': self.chunk_duration,
                'total_chunks': len(chunk_metadata),
                'chunks': chunk_metadata,
                'processed_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save noise analysis summary
        noise_summary_file = self.noise_analysis_dir / f"{base_name}_noise_summary.json"
        with open(noise_summary_file, 'w') as f:
            json.dump({
                'original_file': video_file,
                'chunk_duration': self.chunk_duration,
                'total_chunks': len(noise_analysis),
                'analysis': noise_analysis,
                'statistics': self._calculate_noise_statistics(noise_analysis)
            }, f, indent=2)
        
        print(f"\n✓ Successfully processed {video_file}")
        print(f"  - Created {len(audio_chunks)} audio chunks")
        print(f"  - Created {len(video_chunks)} video chunks")
        print(f"  - Analyzed {len(noise_analysis)} audio segments")
        
        return {
            'success': True,
            'original_file': video_file,
            'base_metadata': base_metadata,
            'audio_chunks': [str(p) for p in audio_chunks],
            'video_chunks': [str(p) for p in video_chunks],
            'chunk_metadata_file': str(chunk_meta_file),
            'noise_summary_file': str(noise_summary_file),
            'total_chunks': len(chunk_metadata)
        }
    
    def _calculate_noise_statistics(self, noise_analysis: List[Dict]) -> Dict:
        """
        Calculate statistical summary of noise levels.
        
        Args:
            noise_analysis: List of noise analysis results
            
        Returns:
            Dictionary with statistics
        """
        import numpy as np
        
        # Extract A-weighted SPL values
        spl_values = [r.get('a_weighted_spl', 0) for r in noise_analysis 
                     if 'error' not in r and r.get('a_weighted_spl', 0) != 0]
        leq_values = [r.get('leq_a', 0) for r in noise_analysis 
                     if 'error' not in r and r.get('leq_a', 0) != 0]
        
        if not spl_values:
            return {'error': 'No valid noise data'}
        
        stats = {
            'a_weighted_spl': {
                'mean': float(np.mean(spl_values)),
                'median': float(np.median(spl_values)),
                'min': float(np.min(spl_values)),
                'max': float(np.max(spl_values)),
                'std': float(np.std(spl_values)),
                'percentile_10': float(np.percentile(spl_values, 10)),
                'percentile_90': float(np.percentile(spl_values, 90))
            },
            'leq_a': {
                'mean': float(np.mean(leq_values)),
                'median': float(np.median(leq_values)),
                'min': float(np.min(leq_values)),
                'max': float(np.max(leq_values)),
                'std': float(np.std(leq_values))
            }
        }
        
        return stats
    
    def process_all(self, audio_format: str = 'wav', 
                   file_pattern: str = '*.mp4') -> Dict[str, Dict]:
        """
        Process all video files in input directory.
        
        Args:
            audio_format: Audio output format
            file_pattern: File pattern to match
            
        Returns:
            Dictionary of processing results
        """
        video_files = list(self.input_dir.glob(file_pattern))
        
        if not video_files:
            print(f"No files matching '{file_pattern}' found in {self.input_dir}")
            return {}
        
        print(f"\n{'='*70}")
        print(f"INTEGRATED MEDIA PROCESSING PIPELINE")
        print(f"{'='*70}")
        print(f"Files to process: {len(video_files)}")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Audio format: {audio_format}")
        print(f"{'='*70}")
        
        results = {}
        successful = 0
        failed = 0
        
        for video_file in video_files:
            result = self.process_file_complete(video_file.name, audio_format)
            
            if result.get('success'):
                results[video_file.name] = result
                successful += 1
            else:
                failed += 1
        
        # Create master index
        master_index = {
            'processed_date': datetime.now().isoformat(),
            'chunk_duration': self.chunk_duration,
            'total_files': len(video_files),
            'successful': successful,
            'failed': failed,
            'files': results
        }
        
        master_index_file = self.chunk_metadata_dir / 'master_chunk_index.json'
        with open(master_index_file, 'w') as f:
            json.dump(master_index, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Master index: {master_index_file}")
        print(f"{'='*70}\n")
        
        return results
    
    def get_chunk_info(self, original_filename: str, chunk_index: int) -> Dict:
        """
        Get information about a specific chunk.
        
        Args:
            original_filename: Original video filename
            chunk_index: Index of the chunk
            
        Returns:
            Dictionary with chunk information
        """
        base_name = Path(original_filename).stem
        chunk_meta_file = self.chunk_metadata_dir / f"{base_name}_chunks.json"
        
        if not chunk_meta_file.exists():
            return {'error': 'Chunk metadata not found'}
        
        with open(chunk_meta_file, 'r') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        if chunk_index < 0 or chunk_index >= len(chunks):
            return {'error': 'Invalid chunk index'}
        
        return chunks[chunk_index]
    
    def get_noise_summary(self, original_filename: str) -> Dict:
        """
        Get noise analysis summary for a file.
        
        Args:
            original_filename: Original video filename
            
        Returns:
            Noise analysis summary
        """
        base_name = Path(original_filename).stem
        noise_summary_file = self.noise_analysis_dir / f"{base_name}_noise_summary.json"
        
        if not noise_summary_file.exists():
            return {'error': 'Noise summary not found'}
        
        with open(noise_summary_file, 'r') as f:
            return json.load(f)
