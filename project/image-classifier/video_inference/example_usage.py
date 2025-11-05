"""
Example Usage Script for Dual YOLO Video Inference

This script demonstrates basic usage of the video inference system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from batch_processor import process_video_directory


def main():
    """Run example video processing."""
    
    # Configuration
    config = {
        'video_directory': './test_videos',
        'output_directory': './results',
        'standard_model': '../YOLOv8/yolov8n.pt',
        'mocs_model': '../YOLOv8/runs/detect/mocs_yolov8n_train/weights/best.pt',
        'frame_skip': 30,
        'conf_threshold': 0.5,
        'aggregation_method': 'weighted_average',
        'min_detection_percentage': 10.0,
        'construction_threshold': 0.3,
        'device': 'cuda'
    }
    
    print("=" * 80)
    print("DUAL YOLO VIDEO INFERENCE - EXAMPLE RUN")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 80)
    
    # Check if video directory exists
    video_dir = Path(config['video_directory'])
    if not video_dir.exists():
        print(f"\n⚠ Error: Video directory not found: {video_dir}")
        print(f"Please create the directory and add video files.")
        print(f"\nExample:")
        print(f"  mkdir {video_dir}")
        print(f"  # Add some .mp4 files to {video_dir}")
        return
    
    # Check if models exist
    standard_model = Path(config['standard_model'])
    mocs_model = Path(config['mocs_model'])
    
    if not standard_model.exists():
        print(f"\n⚠ Warning: Standard model not found: {standard_model}")
        print(f"The system will attempt to download yolov8n.pt automatically.")
        config['standard_model'] = 'yolov8n.pt'
    
    if not mocs_model.exists():
        print(f"\n⚠ Warning: MOCS model not found: {mocs_model}")
        print(f"Please train the MOCS model first or update the path.")
        print(f"Setting mocs_model to None - construction detection will be disabled.")
        config['mocs_model'] = None
    
    # Process videos
    print("\nStarting video processing...\n")
    
    try:
        results = process_video_directory(
            directory_path=config['video_directory'],
            standard_model_path=config['standard_model'],
            mocs_model_path=config['mocs_model'],
            output_dir=config['output_directory'],
            frame_skip=config['frame_skip'],
            conf_threshold=config['conf_threshold'],
            aggregation_method=config['aggregation_method'],
            min_detection_percentage=config['min_detection_percentage'],
            construction_threshold=config['construction_threshold'],
            device=config['device']
        )
        
        if results:
            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE!")
            print("=" * 80)
            print(f"\n✓ Successfully processed {len(results)} videos")
            print(f"✓ Results saved to: {config['output_directory']}")
            
            # Quick summary
            construction_count = sum(1 for r in results if r.is_construction_site)
            print(f"\n📊 Quick Summary:")
            print(f"   Construction sites detected: {construction_count}/{len(results)}")
            
            if construction_count > 0:
                print(f"\n🏗️  Construction Site Videos:")
                for result in results:
                    if result.is_construction_site:
                        video_name = Path(result.video_path).name
                        print(f"   - {video_name}: {result.construction_confidence:.1%} confidence")
                        if result.construction_equipment:
                            print(f"     Equipment: {', '.join(result.construction_equipment[:3])}")
            
            print("\n" + "=" * 80)
        else:
            print("\n⚠ No videos were processed. Check the video directory.")
    
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
