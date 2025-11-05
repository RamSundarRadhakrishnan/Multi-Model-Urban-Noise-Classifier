"""
Command-line script for processing MP4 files.
Separates audio and video while maintaining temporal alignment.
"""

import argparse
import sys
from pathlib import Path
from media_processor import MediaProcessor
from media_utils import (
    check_dependencies,
    print_dependency_status,
    print_processing_summary,
    get_processed_files_summary
)


def main():
    parser = argparse.ArgumentParser(
        description='Process MP4 files: separate audio and video with temporal alignment'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./input_data_buffer',
        help='Input directory containing MP4 files (default: ./input_data_buffer)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./processed_media',
        help='Output directory for processed files (default: ./processed_media)'
    )
    
    parser.add_argument(
        '-f', '--format',
        type=str,
        default='wav',
        choices=['wav', 'mp3', 'aac', 'flac'],
        help='Audio output format (default: wav)'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.mp4',
        help='File pattern to match (default: *.mp4)'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary of previously processed files'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    print("Checking dependencies...\n")
    dependencies = check_dependencies()
    print_dependency_status(dependencies)
    
    if not all(dependencies.values()):
        print("\n❌ ERROR: Missing required dependencies!")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        sys.exit(1)
    
    if args.check_deps:
        print("\n✓ All dependencies installed!")
        sys.exit(0)
    
    # Show summary if requested
    if args.summary:
        metadata_dir = Path(args.output) / 'metadata'
        if metadata_dir.exists():
            summary = get_processed_files_summary(str(metadata_dir))
            print_processing_summary(summary)
        else:
            print(f"\nNo processed files found in {metadata_dir}")
        sys.exit(0)
    
    # Initialize processor
    print(f"\nInitializing MediaProcessor...")
    print(f"  Input:  {Path(args.input).absolute()}")
    print(f"  Output: {Path(args.output).absolute()}")
    print(f"  Format: {args.format}")
    print(f"  Pattern: {args.pattern}\n")
    
    processor = MediaProcessor(
        input_dir=args.input,
        output_base_dir=args.output
    )
    
    # Process files
    try:
        results = processor.process_all(
            audio_format=args.format,
            file_pattern=args.pattern
        )
        
        if results:
            print("\n✓ Processing completed successfully!")
            print(f"\nOutput directories:")
            print(f"  Audio:    {processor.audio_dir}")
            print(f"  Video:    {processor.video_dir}")
            print(f"  Metadata: {processor.metadata_dir}")
            
            # Show summary
            summary = get_processed_files_summary(str(processor.metadata_dir))
            print_processing_summary(summary)
            
            sys.exit(0)
        else:
            print("\n⚠️  No files were processed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
