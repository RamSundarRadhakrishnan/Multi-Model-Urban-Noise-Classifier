"""
System Verification Test

Run this script to verify that all modules are properly installed and accessible.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from video_frame_extractor import VideoFrameExtractor
        print("✓ video_frame_extractor")
    except ImportError as e:
        print(f"✗ video_frame_extractor: {e}")
        return False
    
    try:
        from dual_yolo_inference import DualYOLOInference, CONSTRUCTION_EQUIPMENT
        print("✓ dual_yolo_inference")
    except ImportError as e:
        print(f"✗ dual_yolo_inference: {e}")
        return False
    
    try:
        from output_fusion import OutputFusion, create_fusion_engine, AggregationMethod
        print("✓ output_fusion")
    except ImportError as e:
        print(f"✗ output_fusion: {e}")
        return False
    
    try:
        from batch_processor import BatchVideoProcessor, process_video_directory
        print("✓ batch_processor")
    except ImportError as e:
        print(f"✗ batch_processor: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('ultralytics', 'ultralytics'),
        ('torch', 'torch'),
    ]
    
    all_installed = True
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def test_construction_equipment():
    """Test that construction equipment classes are defined."""
    print("\nTesting construction equipment classes...")
    
    from dual_yolo_inference import CONSTRUCTION_EQUIPMENT
    
    print(f"✓ {len(CONSTRUCTION_EQUIPMENT)} construction equipment classes defined")
    print(f"  Classes: {', '.join(sorted(list(CONSTRUCTION_EQUIPMENT)[:5]))}...")
    
    return True


def test_aggregation_methods():
    """Test that all aggregation methods are available."""
    print("\nTesting aggregation methods...")
    
    from output_fusion import AggregationMethod
    
    methods = [
        AggregationMethod.MAX_CONFIDENCE,
        AggregationMethod.AVERAGE_CONFIDENCE,
        AggregationMethod.MAJORITY_VOTE,
        AggregationMethod.WEIGHTED_AVERAGE,
        AggregationMethod.THRESHOLD_PERCENTAGE
    ]
    
    print(f"✓ {len(methods)} aggregation methods available")
    for method in methods:
        print(f"  - {method.value}")
    
    return True


def test_file_structure():
    """Test that all required files are present."""
    print("\nTesting file structure...")
    
    required_files = [
        'video_frame_extractor.py',
        'dual_yolo_inference.py',
        'output_fusion.py',
        'batch_processor.py',
        '__init__.py',
        'README.md',
        'QUICKSTART.md',
        'requirements.txt',
        'example_usage.py',
        'dual_yolo_inference_pipeline.ipynb',
    ]
    
    base_path = Path(__file__).parent
    all_present = True
    
    for filename in required_files:
        file_path = base_path / filename
        if file_path.exists():
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} - MISSING")
            all_present = False
    
    return all_present


def main():
    """Run all tests."""
    print("=" * 80)
    print("DUAL YOLO VIDEO INFERENCE - SYSTEM VERIFICATION")
    print("=" * 80)
    
    results = []
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test construction equipment
    results.append(("Construction Equipment", test_construction_equipment()))
    
    # Test aggregation methods
    results.append(("Aggregation Methods", test_aggregation_methods()))
    
    # Test file structure
    results.append(("File Structure", test_file_structure()))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run 'python example_usage.py' to process videos")
        print("  2. Or open 'dual_yolo_inference_pipeline.ipynb' for interactive exploration")
        print("  3. Read QUICKSTART.md for a quick introduction")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("\nTo fix dependency issues, run:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
