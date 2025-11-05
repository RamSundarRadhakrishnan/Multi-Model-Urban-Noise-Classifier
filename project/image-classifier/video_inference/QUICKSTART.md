# Quick Start Guide - Dual YOLO Video Inference

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd d:\Projects\UrbanNoiseClassifier\project\image-classifier\video_inference
pip install -r requirements.txt
```

### Step 2: Prepare Your Videos

Create a test videos directory and add your video files:

```bash
mkdir test_videos
# Copy some .mp4 or .avi files to test_videos/
```

### Step 3: Run the Example Script

```bash
python example_usage.py
```

This will:

- ✅ Process all videos in `test_videos/`
- ✅ Run both standard YOLO and MOCS detection
- ✅ Identify construction sites automatically
- ✅ Save results to `results/` directory

### Step 4: View Results

Check the `results/` directory for:

- `video_inference_summary_*.csv` - Quick overview
- `video_inference_results_*.json` - Detailed results
- `detailed_reports_*/` - Per-video detailed reports

---

## 📓 Using the Jupyter Notebook

For interactive exploration and visualization:

```bash
jupyter notebook dual_yolo_inference_pipeline.ipynb
```

The notebook includes:

- 📊 Data visualization examples
- 🎯 Single video processing walkthrough
- 📁 Batch processing demonstrations
- 📈 Results analysis and comparison

---

## 💻 Python API Usage

### Minimal Example

```python
from batch_processor import process_video_directory

results = process_video_directory(
    directory_path='./test_videos',
    standard_model_path='yolov8n.pt',
    mocs_model_path='path/to/mocs_model.pt',
    output_dir='./results'
)

for result in results:
    print(f"{result.video_path}:")
    print(f"  Construction site: {result.is_construction_site}")
    print(f"  Confidence: {result.construction_confidence:.2%}")
```

### Detailed Control

```python
from video_frame_extractor import VideoFrameExtractor
from dual_yolo_inference import DualYOLOInference
from output_fusion import create_fusion_engine

# Initialize
extractor = VideoFrameExtractor(frame_skip=30)
inference = DualYOLOInference(
    standard_model_path='yolov8n.pt',
    mocs_model_path='mocs_best.pt',
    conf_threshold=0.5,
    device='cuda'
)
fusion = create_fusion_engine(aggregation_method='weighted_average')

# Process
frames = list(extractor.extract_frames('video.mp4'))
frame_results = inference.infer_frames(frames)
video_result = fusion.fuse_video_results('video.mp4', frame_results)

# Results
print(f"Construction: {video_result.is_construction_site}")
print(f"Equipment: {video_result.construction_equipment}")
print(f"Top objects: {[d.class_name for d in video_result.standard_detections[:5]]}")
```

---

## ⚙️ Configuration Options

### Processing Speed

**Fast Mode** (0.5 fps):

```python
results = process_video_directory(
    directory_path='./videos',
    frame_skip=60,  # Process fewer frames
    device='cuda'
)
```

**Accuracy Mode** (2 fps):

```python
results = process_video_directory(
    directory_path='./videos',
    frame_skip=15,  # Process more frames
    conf_threshold=0.6,  # Higher confidence
    device='cuda'
)
```

### Aggregation Methods

```python
# Maximum confidence (best single detection)
aggregation_method='max_confidence'

# Average confidence (stable across frames)
aggregation_method='average_confidence'

# Weighted by frequency + confidence (recommended)
aggregation_method='weighted_average'

# Majority vote (most consistent)
aggregation_method='majority_vote'
```

### Construction Detection Sensitivity

```python
# More sensitive (detects more construction sites)
construction_threshold=0.1  # Equipment in 10% of frames

# Balanced (default)
construction_threshold=0.3  # Equipment in 30% of frames

# Conservative (high confidence only)
construction_threshold=0.5  # Equipment in 50% of frames
```

---

## 📊 Understanding Output

### JSON Output Structure

```json
{
  "video_path": "construction_site.mp4",
  "total_frames": 100,
  "is_construction_site": true,
  "construction_confidence": 0.85,
  "construction_equipment": ["Excavator", "Truck"],
  "standard_detections": [
    {
      "class_name": "person",
      "confidence": 0.92,
      "detection_count": 78,
      "detection_percentage": 78.0,
      "model_source": "standard"
    }
  ]
}
```

### CSV Summary Fields

- **Video**: Filename
- **Frames**: Number of frames analyzed
- **Construction Site**: Yes/No
- **Construction Confidence**: Percentage
- **Equipment**: Detected construction equipment
- **Top Detection**: Most confident object detected
- **Detection Confidence**: Confidence of top detection

---

## 🔧 Troubleshooting

### "CUDA out of memory"

```python
# Solution 1: Use CPU
device='cpu'

# Solution 2: Process fewer frames
frame_skip=60

# Solution 3: Use smaller model
standard_model_path='yolov8n.pt'  # nano (smallest)
```

### "No construction sites detected"

```python
# Solution 1: Lower threshold
construction_threshold=0.1

# Solution 2: Check MOCS model path
mocs_model_path='correct/path/to/best.pt'

# Solution 3: Verify equipment in video
# Equipment must be visible, not just workers
```

### "Too many false positives"

```python
# Solution 1: Increase confidence
conf_threshold=0.7

# Solution 2: Increase min detection percentage
min_detection_percentage=20.0

# Solution 3: Use stricter aggregation
aggregation_method='max_confidence'
```

---

## 📁 Project Structure

```
video_inference/
├── README.md                      # Full documentation
├── QUICKSTART.md                  # This file
├── requirements.txt               # Dependencies
├── example_usage.py               # Example script
├── dual_yolo_inference_pipeline.ipynb  # Interactive notebook
├── video_frame_extractor.py       # Frame extraction
├── dual_yolo_inference.py         # Dual YOLO inference
├── output_fusion.py               # Result aggregation
├── batch_processor.py             # Batch processing
├── __init__.py                    # Package initialization
├── test_videos/                   # Your input videos
└── results/                       # Output directory
```

---

## 🎯 Common Use Cases

### Use Case 1: Identify Construction Sites in City Footage

```python
results = process_video_directory(
    directory_path='./city_footage',
    mocs_model_path='mocs_best.pt',
    construction_threshold=0.3,
    output_dir='./construction_sites'
)

# Filter construction sites
construction_videos = [r for r in results if r.is_construction_site]
```

### Use Case 2: General Object Detection in Videos

```python
results = process_video_directory(
    directory_path='./videos',
    mocs_model_path=None,  # Skip MOCS model
    aggregation_method='weighted_average',
    output_dir='./object_detections'
)

# Get all detected objects
all_objects = set()
for result in results:
    for detection in result.standard_detections:
        all_objects.add(detection.class_name)
```

### Use Case 3: Fast Screening of Large Video Collection

```python
results = process_video_directory(
    directory_path='./large_collection',
    frame_skip=120,  # Very fast, low detail
    conf_threshold=0.4,
    device='cuda'
)
```

---

## 📚 Next Steps

1. **Read the full README.md** for comprehensive documentation
2. **Explore the Jupyter notebook** for interactive examples
3. **Customize configuration** for your specific use case
4. **Integrate into your pipeline** using the Python API

---

## 🆘 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the full README.md
3. Examine example outputs in the notebook
4. Open an issue in the repository

---

## ✨ Tips for Best Results

1. **Use GPU**: 10-50x faster than CPU
2. **Adjust frame_skip**: Balance speed vs accuracy
3. **Tune confidence**: Higher = fewer false positives
4. **Check your models**: Ensure MOCS model is trained properly
5. **Validate results**: Review a few videos manually to calibrate settings

---

**Ready to process videos? Run `python example_usage.py` to get started!** 🚀
