"""
Chunking Module
Handles splitting audio and video into time-synchronized chunks.
"""

import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import json
import math


class MediaChunker:
    """
    Split audio and video files into synchronized chunks.
    """
    
    def __init__(self, chunk_duration: float = 10.0):
        """
        Initialize MediaChunker.
        
        Args:
            chunk_duration: Duration of each chunk in seconds (default: 10.0)
        """
        self.chunk_duration = chunk_duration
    
    def get_media_duration(self, filepath: Path) -> float:
        """
        Get duration of media file using ffprobe.
        
        Args:
            filepath: Path to media file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                str(filepath)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            duration = float(data.get('format', {}).get('duration', 0))
            return duration
        except Exception as e:
            print(f"Error getting duration: {e}")
            return 0.0
    
    def calculate_chunks(self, duration: float) -> List[Tuple[float, float]]:
        """
        Calculate chunk start and end times.
        
        Args:
            duration: Total duration in seconds
            
        Returns:
            List of (start_time, end_time) tuples
        """
        chunks = []
        num_chunks = math.ceil(duration / self.chunk_duration)
        
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            end_time = min((i + 1) * self.chunk_duration, duration)
            chunks.append((start_time, end_time))
        
        return chunks
    
    def chunk_audio(self, input_path: Path, output_dir: Path, 
                   base_name: str = None) -> List[Path]:
        """
        Split audio file into chunks.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory for output chunks
            base_name: Base name for output files (default: input filename)
            
        Returns:
            List of output file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = input_path.stem
        
        # Get duration
        duration = self.get_media_duration(input_path)
        
        if duration <= 0:
            print(f"Could not get duration for {input_path}")
            return []
        
        # Calculate chunks
        chunks = self.calculate_chunks(duration)
        output_paths = []
        
        for i, (start, end) in enumerate(chunks):
            chunk_duration = end - start
            output_path = output_dir / f"{base_name}_chunk_{i:04d}.wav"
            
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start),
                '-t', str(chunk_duration),
                '-acodec', 'copy',
                '-y',
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                output_paths.append(output_path)
            except subprocess.CalledProcessError as e:
                print(f"Error creating chunk {i}: {e}")
        
        return output_paths
    
    def chunk_video(self, input_path: Path, output_dir: Path, 
                   base_name: str = None) -> List[Path]:
        """
        Split video file into chunks.
        
        Args:
            input_path: Path to input video file
            output_dir: Directory for output chunks
            base_name: Base name for output files (default: input filename)
            
        Returns:
            List of output file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            base_name = input_path.stem
        
        # Get duration
        duration = self.get_media_duration(input_path)
        
        if duration <= 0:
            print(f"Could not get duration for {input_path}")
            return []
        
        # Calculate chunks
        chunks = self.calculate_chunks(duration)
        output_paths = []
        
        for i, (start, end) in enumerate(chunks):
            chunk_duration = end - start
            output_path = output_dir / f"{base_name}_chunk_{i:04d}.mp4"
            
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-ss', str(start),
                '-t', str(chunk_duration),
                '-vcodec', 'copy',
                '-an',  # No audio
                '-y',
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                output_paths.append(output_path)
            except subprocess.CalledProcessError as e:
                print(f"Error creating chunk {i}: {e}")
        
        return output_paths
    
    def chunk_media_pair(self, audio_path: Path, video_path: Path,
                        audio_output_dir: Path, video_output_dir: Path,
                        base_name: str = None) -> Tuple[List[Path], List[Path]]:
        """
        Split both audio and video into synchronized chunks.
        
        Args:
            audio_path: Path to audio file
            video_path: Path to video file
            audio_output_dir: Output directory for audio chunks
            video_output_dir: Output directory for video chunks
            base_name: Base name for output files
            
        Returns:
            Tuple of (audio_chunk_paths, video_chunk_paths)
        """
        print(f"Chunking {base_name or audio_path.stem}...")
        
        audio_chunks = self.chunk_audio(audio_path, audio_output_dir, base_name)
        video_chunks = self.chunk_video(video_path, video_output_dir, base_name)
        
        print(f"  Created {len(audio_chunks)} audio chunks and {len(video_chunks)} video chunks")
        
        return audio_chunks, video_chunks
    
    def create_chunk_metadata(self, audio_chunks: List[Path], video_chunks: List[Path],
                            original_file: str, base_name: str) -> List[Dict]:
        """
        Create metadata for chunks.
        
        Args:
            audio_chunks: List of audio chunk paths
            video_chunks: List of video chunk paths
            original_file: Original filename
            base_name: Base name used for chunks
            
        Returns:
            List of chunk metadata dictionaries
        """
        chunk_metadata = []
        
        for i, (audio_chunk, video_chunk) in enumerate(zip(audio_chunks, video_chunks)):
            start_time = i * self.chunk_duration
            end_time = start_time + self.chunk_duration
            
            metadata = {
                'chunk_index': i,
                'original_file': original_file,
                'audio_chunk': str(audio_chunk.name),
                'video_chunk': str(video_chunk.name),
                'start_time': start_time,
                'end_time': end_time,
                'duration': self.chunk_duration,
                'temporal_alignment': {
                    'synchronized': True,
                    'chunk_duration': self.chunk_duration
                }
            }
            
            chunk_metadata.append(metadata)
        
        return chunk_metadata
