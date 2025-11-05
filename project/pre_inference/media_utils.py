"""
Utility functions for media processing and analysis.
"""

import json
from pathlib import Path
from typing import Dict, List
import subprocess


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dictionary with dependency names and their availability status
    """
    dependencies = {}
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        dependencies['ffmpeg'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        dependencies['ffmpeg'] = False
    
    # Check ffprobe
    try:
        subprocess.run(['ffprobe', '-version'], 
                      capture_output=True, check=True)
        dependencies['ffprobe'] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        dependencies['ffprobe'] = False
    
    return dependencies


def print_dependency_status(dependencies: Dict[str, bool]):
    """
    Print the status of dependencies in a formatted way.
    
    Args:
        dependencies: Dictionary of dependency statuses
    """
    print("Dependency Check:")
    print("-" * 30)
    for dep, status in dependencies.items():
        status_str = "✓ Installed" if status else "✗ Not Found"
        print(f"{dep:15} {status_str}")
    print("-" * 30)


def load_master_index(metadata_dir: str) -> Dict:
    """
    Load the master index file.
    
    Args:
        metadata_dir: Directory containing metadata files
        
    Returns:
        Master index dictionary
    """
    index_path = Path(metadata_dir) / 'master_index.json'
    
    if not index_path.exists():
        return {}
    
    with open(index_path, 'r') as f:
        return json.load(f)


def get_processed_files_summary(metadata_dir: str) -> Dict:
    """
    Get a summary of all processed files.
    
    Args:
        metadata_dir: Directory containing metadata files
        
    Returns:
        Summary dictionary
    """
    master_index = load_master_index(metadata_dir)
    
    if not master_index:
        return {
            'total_files': 0,
            'files': []
        }
    
    summary = {
        'total_files': master_index.get('total_files', 0),
        'successful': master_index.get('successful', 0),
        'failed': master_index.get('failed', 0),
        'processed_date': master_index.get('processed_date', 'Unknown'),
        'files': []
    }
    
    files_data = master_index.get('files', {})
    for filename, metadata in files_data.items():
        summary['files'].append({
            'filename': filename,
            'duration': metadata.get('temporal_alignment', {}).get('duration', 'N/A'),
            'audio_file': metadata.get('audio_file', 'N/A'),
            'video_file': metadata.get('video_file', 'N/A')
        })
    
    return summary


def print_processing_summary(summary: Dict):
    """
    Print a formatted summary of processed files.
    
    Args:
        summary: Summary dictionary from get_processed_files_summary
    """
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total Files Processed: {summary.get('total_files', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Processed Date: {summary.get('processed_date', 'Unknown')}")
    print("=" * 60)
    
    if summary.get('files'):
        print("\nProcessed Files:")
        print("-" * 60)
        for i, file_info in enumerate(summary['files'], 1):
            print(f"\n{i}. {file_info['filename']}")
            print(f"   Duration: {file_info['duration']} seconds")
            print(f"   Audio: {file_info['audio_file']}")
            print(f"   Video: {file_info['video_file']}")
    else:
        print("\nNo files have been processed yet.")
    
    print("\n" + "=" * 60 + "\n")


def validate_temporal_alignment(metadata: Dict) -> bool:
    """
    Validate that temporal alignment data is complete.
    
    Args:
        metadata: Metadata dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['duration', 'frame_rate', 'sample_rate']
    temporal_data = metadata.get('temporal_alignment', {})
    
    for field in required_fields:
        if field not in temporal_data or temporal_data[field] is None:
            return False
    
    return True


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def get_file_list(directory: str, pattern: str = '*.mp4') -> List[Path]:
    """
    Get list of files matching pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for files
        
    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        return []
    
    return sorted(list(dir_path.glob(pattern)))
