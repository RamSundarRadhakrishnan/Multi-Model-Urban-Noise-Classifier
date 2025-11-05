"""
Media Processor Module
Handles separation of audio and video from MP4 files while maintaining temporal alignment.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import subprocess


class MediaProcessor:
    """
    Process MP4 files to separate audio and video streams while maintaining
    temporal alignment through metadata mapping.
    """
    
    def __init__(self, input_dir: str, output_base_dir: str = None):
        """
        Initialize the MediaProcessor.
        
        Args:
            input_dir: Directory containing input MP4 files
            output_base_dir: Base directory for output files. If None, creates
                           'processed_media' in the same parent directory as input_dir
        """
        self.input_dir = Path(input_dir)
        
        if output_base_dir is None:
            self.output_base_dir = self.input_dir.parent / 'processed_media'
        else:
            self.output_base_dir = Path(output_base_dir)
        
        # Create output subdirectories
        self.audio_dir = self.output_base_dir / 'audio'
        self.video_dir = self.output_base_dir / 'video'
        self.metadata_dir = self.output_base_dir / 'metadata'
        
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary output directories."""
        for directory in [self.audio_dir, self.video_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _check_ffmpeg(self) -> bool:
        """
        Check if ffmpeg is installed and accessible.
        
        Returns:
            bool: True if ffmpeg is available, False otherwise
        """
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _get_media_info(self, video_path: Path) -> Dict:
        """
        Extract media information using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing media information
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error getting media info: {e}")
            return {}
    
    def extract_audio(self, video_path: Path, output_path: Path, 
                     audio_format: str = 'wav') -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            audio_format: Output audio format (wav, mp3, aac, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le' if audio_format == 'wav' else 'copy',
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from {video_path.name}: {e}")
            return False
    
    def extract_video(self, video_path: Path, output_path: Path) -> bool:
        """
        Extract video stream without audio.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output video file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-an',  # No audio
                '-vcodec', 'copy',  # Copy video codec
                '-y',  # Overwrite output file
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting video from {video_path.name}: {e}")
            return False
    
    def create_metadata(self, video_path: Path, audio_path: Path, 
                       video_only_path: Path, media_info: Dict) -> Dict:
        """
        Create metadata mapping for temporal alignment.
        
        Args:
            video_path: Original video file path
            audio_path: Extracted audio file path
            video_only_path: Extracted video-only file path
            media_info: Media information from ffprobe
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'original_file': str(video_path.name),
            'original_path': str(video_path),
            'audio_file': str(audio_path.name),
            'audio_path': str(audio_path),
            'video_file': str(video_only_path.name),
            'video_path': str(video_only_path),
            'processed_timestamp': datetime.now().isoformat(),
            'temporal_alignment': {
                'start_time': 0.0,
                'duration': None,
                'frame_rate': None,
                'sample_rate': None
            }
        }
        
        # Extract duration and other info from media_info
        if 'format' in media_info:
            metadata['temporal_alignment']['duration'] = float(
                media_info['format'].get('duration', 0)
            )
        
        # Extract video and audio stream information
        if 'streams' in media_info:
            for stream in media_info['streams']:
                if stream['codec_type'] == 'video':
                    metadata['temporal_alignment']['frame_rate'] = stream.get('r_frame_rate', 'N/A')
                    metadata['video_codec'] = stream.get('codec_name', 'N/A')
                    metadata['video_width'] = stream.get('width', 'N/A')
                    metadata['video_height'] = stream.get('height', 'N/A')
                elif stream['codec_type'] == 'audio':
                    metadata['temporal_alignment']['sample_rate'] = stream.get('sample_rate', 'N/A')
                    metadata['audio_codec'] = stream.get('codec_name', 'N/A')
                    metadata['audio_channels'] = stream.get('channels', 'N/A')
        
        return metadata
    
    def process_file(self, video_file: str, audio_format: str = 'wav') -> Tuple[bool, Dict]:
        """
        Process a single video file.
        
        Args:
            video_file: Name of the video file in input directory
            audio_format: Output audio format (default: wav)
            
        Returns:
            Tuple of (success: bool, metadata: dict)
        """
        video_path = self.input_dir / video_file
        
        if not video_path.exists():
            print(f"File not found: {video_path}")
            return False, {}
        
        # Generate output filenames
        base_name = video_path.stem
        audio_output = self.audio_dir / f"{base_name}.{audio_format}"
        video_output = self.video_dir / f"{base_name}_video_only.mp4"
        metadata_output = self.metadata_dir / f"{base_name}_metadata.json"
        
        # Get media information
        media_info = self._get_media_info(video_path)
        
        # Extract audio
        print(f"Extracting audio from {video_file}...")
        audio_success = self.extract_audio(video_path, audio_output, audio_format)
        
        # Extract video
        print(f"Extracting video from {video_file}...")
        video_success = self.extract_video(video_path, video_output)
        
        if audio_success and video_success:
            # Create and save metadata
            metadata = self.create_metadata(video_path, audio_output, 
                                           video_output, media_info)
            
            with open(metadata_output, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Successfully processed {video_file}")
            return True, metadata
        else:
            print(f"✗ Failed to process {video_file}")
            return False, {}
    
    def process_all(self, audio_format: str = 'wav', 
                   file_pattern: str = '*.mp4') -> Dict[str, Dict]:
        """
        Process all video files in the input directory.
        
        Args:
            audio_format: Output audio format (default: wav)
            file_pattern: Glob pattern for files to process (default: *.mp4)
            
        Returns:
            Dictionary mapping filenames to their metadata
        """
        if not self._check_ffmpeg():
            raise RuntimeError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg to use this processor."
            )
        
        video_files = list(self.input_dir.glob(file_pattern))
        
        if not video_files:
            print(f"No files matching '{file_pattern}' found in {self.input_dir}")
            return {}
        
        print(f"Found {len(video_files)} file(s) to process\n")
        
        results = {}
        successful = 0
        failed = 0
        
        for video_file in video_files:
            success, metadata = self.process_file(video_file.name, audio_format)
            
            if success:
                results[video_file.name] = metadata
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*50}\n")
        
        # Save master index
        index_path = self.metadata_dir / 'master_index.json'
        with open(index_path, 'w') as f:
            json.dump({
                'processed_date': datetime.now().isoformat(),
                'total_files': len(video_files),
                'successful': successful,
                'failed': failed,
                'files': results
            }, f, indent=2)
        
        print(f"Master index saved to: {index_path}")
        
        return results
    
    def get_aligned_pair(self, original_filename: str) -> Tuple[Path, Path, Dict]:
        """
        Get the audio and video paths for a processed file along with metadata.
        
        Args:
            original_filename: Name of the original video file
            
        Returns:
            Tuple of (audio_path, video_path, metadata)
        """
        base_name = Path(original_filename).stem
        
        audio_path = self.audio_dir / f"{base_name}.wav"
        video_path = self.video_dir / f"{base_name}_video_only.mp4"
        metadata_path = self.metadata_dir / f"{base_name}_metadata.json"
        
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return audio_path, video_path, metadata
