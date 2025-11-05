"""
Example usage of the media processing pipeline.
This script demonstrates various use cases.
"""

from pathlib import Path
from media_processor import MediaProcessor
from media_utils import check_dependencies, print_dependency_status
import json


def example_basic_processing():
    """Basic example: Process all MP4 files in default directory."""
    print("="*70)
    print("EXAMPLE 1: Basic Processing")
    print("="*70)
    
    # Check dependencies first
    deps = check_dependencies()
    print_dependency_status(deps)
    
    if not all(deps.values()):
        print("Please install ffmpeg first!")
        return
    
    # Initialize processor
    processor = MediaProcessor('./input_data_buffer')
    
    # Process all files
    results = processor.process_all(audio_format='wav')
    
    print(f"\nProcessed {len(results)} file(s)")


def example_custom_directories():
    """Example: Use custom input and output directories."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Directories")
    print("="*70)
    
    processor = MediaProcessor(
        input_dir='./custom_input',
        output_base_dir='./custom_output'
    )
    
    print(f"Input:  {processor.input_dir}")
    print(f"Output: {processor.output_base_dir}")
    
    # Process with MP3 audio format
    # results = processor.process_all(audio_format='mp3')


def example_single_file_processing():
    """Example: Process a single specific file."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Single File Processing")
    print("="*70)
    
    processor = MediaProcessor('./input_data_buffer')
    
    # Process one file
    filename = 'sample_video.mp4'  # Replace with actual filename
    success, metadata = processor.process_file(filename, audio_format='wav')
    
    if success:
        print(f"✓ Successfully processed {filename}")
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
    else:
        print(f"✗ File not found: {filename}")


def example_get_aligned_pair():
    """Example: Retrieve aligned audio/video pair for a processed file."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Get Aligned Audio/Video Pair")
    print("="*70)
    
    processor = MediaProcessor('./input_data_buffer')
    
    # Get aligned pair for a specific file
    original_file = 'sample_video.mp4'  # Replace with actual filename
    audio_path, video_path, metadata = processor.get_aligned_pair(original_file)
    
    print(f"Original: {original_file}")
    print(f"Audio:    {audio_path}")
    print(f"Video:    {video_path}")
    
    if metadata:
        duration = metadata.get('temporal_alignment', {}).get('duration', 0)
        print(f"Duration: {duration:.2f} seconds")


def example_batch_with_pattern():
    """Example: Process only files matching a specific pattern."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Process Files Matching Pattern")
    print("="*70)
    
    processor = MediaProcessor('./input_data_buffer')
    
    # Process only files starting with 'scene'
    results = processor.process_all(
        audio_format='wav',
        file_pattern='scene*.mp4'
    )
    
    print(f"Processed {len(results)} file(s) matching pattern 'scene*.mp4'")


def example_verify_temporal_alignment():
    """Example: Verify temporal alignment of processed files."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Verify Temporal Alignment")
    print("="*70)
    
    processor = MediaProcessor('./input_data_buffer')
    
    # Load master index
    metadata_dir = processor.metadata_dir
    master_index_path = metadata_dir / 'master_index.json'
    
    if master_index_path.exists():
        with open(master_index_path, 'r') as f:
            master_index = json.load(f)
        
        print("Temporal Alignment Information:")
        print("-" * 70)
        
        for filename, metadata in master_index.get('files', {}).items():
            temporal = metadata.get('temporal_alignment', {})
            print(f"\n{filename}:")
            print(f"  Duration:     {temporal.get('duration', 'N/A')} seconds")
            print(f"  Frame Rate:   {temporal.get('frame_rate', 'N/A')}")
            print(f"  Sample Rate:  {temporal.get('sample_rate', 'N/A')} Hz")
            print(f"  Start Time:   {temporal.get('start_time', 'N/A')} seconds")
    else:
        print("No processed files found. Run processing first.")


def example_different_audio_formats():
    """Example: Process with different audio formats."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Different Audio Formats")
    print("="*70)
    
    processor = MediaProcessor('./input_data_buffer')
    
    # WAV format (highest quality, larger files)
    # results_wav = processor.process_all(audio_format='wav')
    
    # MP3 format (compressed, smaller files)
    # results_mp3 = processor.process_all(audio_format='mp3')
    
    # AAC format (modern compression)
    # results_aac = processor.process_all(audio_format='aac')
    
    print("Supported formats: wav, mp3, aac, flac")
    print("Default: wav (PCM 16-bit, 44100 Hz, stereo)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("MEDIA PROCESSING PIPELINE - USAGE EXAMPLES")
    print("="*70 + "\n")
    
    # Run examples (comment out as needed)
    example_basic_processing()
    example_custom_directories()
    example_single_file_processing()
    example_get_aligned_pair()
    example_batch_with_pattern()
    example_verify_temporal_alignment()
    example_different_audio_formats()
    
    print("\n" + "="*70)
    print("For more information, see README.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
