# Quick Start Guide - Integrated Noise Classification Pipeline

## 🚀 Get Started in 5 Minutes

### Step 1: Verify Dependencies

```powershell
# Check if ffmpeg is installed
ffmpeg -version

# If not installed:
choco install ffmpeg  # Windows with Chocolatey
```

### Step 2: Install Python Packages

```powershell
cd d:\Projects\UrbanNoiseClassifier\project\inference
pip install -r requirements.txt
```

### Step 3: Prepare Input Data

```powershell
# Create input directory (if not exists)
mkdir ..\pre_inference\input_data_buffer -Force

# Copy your MP4 files to the input directory
# Example:
# Copy-Item "C:\path\to\your\videos\*.mp4" -Destination "..\pre_inference\input_data_buffer"
```

### Step 4: Verify Model Files

Ensure you have the following models:

**Audio Model:**

- `../audio-classifier/speechbrain-classifier/pretrained_urbansound8k/`
- `../audio-classifier/speechbrain-classifier/fine_tuned_urbansound8k_8class.pth`

**Video Models:**

- `../image-classifier/YOLOv8/yolov8n.pt`
- `../image-classifier/YOLOv8/runs/detect/mocs_yolov8n_train/weights/best.pt`

### Step 5: Run the Pipeline

**Option A: Using Jupyter Notebook**

```powershell
# Start Jupyter
jupyter notebook inference.ipynb

# Then run all cells in the notebook
```

**Option B: Using VS Code**

```powershell
# Open in VS Code
code inference.ipynb

# Run all cells using the notebook interface
```

## 📊 Understanding the Output

After running the pipeline, check the `output/events/` directory:

### Generated Files

```
output/events/
├── json/
│   ├── events_YYYYMMDD_HHMMSS.json          # All events
│   └── violations_YYYYMMDD_HHMMSS.json      # Violations only
├── csv/
│   ├── events_YYYYMMDD_HHMMSS.csv           # Summary CSV
│   └── events_detailed_YYYYMMDD_HHMMSS.csv  # Detailed CSV
└── summaries/
    └── summary_YYYYMMDD_HHMMSS.json         # Statistical summary
```

### Key Metrics to Check

1. **Compliance Rate**: Percentage of events meeting regulations
2. **Violation Count**: Number of events exceeding limits
3. **Severity Breakdown**: Distribution of minor/moderate/severe violations
4. **Average Noise Level**: Mean dB(A) across all events

## ⚙️ Common Configurations

### For Residential Area Monitoring

```python
CONFIG = {
    'default_zone': NoiseZone.RESIDENTIAL,
    'chunk_duration': 10.0,
    'skip_preprocessing': False,  # Process new media
}
```

### For Construction Site Monitoring

```python
CONFIG = {
    'default_zone': NoiseZone.INDUSTRIAL,
    'construction_threshold': 0.6,  # More sensitive
    'frame_skip': 15,  # More detailed video analysis
}
```

### For Silence Zone (Hospital/School Area)

```python
CONFIG = {
    'default_zone': NoiseZone.SILENCE,
    'chunk_duration': 5.0,  # Shorter chunks for finer detail
    'yolo_conf_threshold': 0.4,  # Lower threshold
}
```

### For Quick Testing (Skip Heavy Processing)

```python
CONFIG = {
    'skip_preprocessing': True,  # Use existing chunks
    'skip_audio_inference': True,  # Use existing results
    'skip_video_inference': True,  # Use existing results
    # Only re-run classification with different zone
    'default_zone': NoiseZone.SILENCE,
}
```

## 🔧 Troubleshooting

### Problem: "No module named 'integrated_processor'"

**Solution:** Make sure you're running from the correct directory

```powershell
cd d:\Projects\UrbanNoiseClassifier\project\inference
```

### Problem: CUDA out of memory

**Solution:** Reduce batch sizes and frame processing

```python
CONFIG = {
    'audio_batch_size': 4,  # Reduce from 8
    'frame_skip': 60,  # Increase from 30
    'device': 'cpu',  # Use CPU instead of GPU
}
```

### Problem: "ffmpeg not found"

**Solution:** Install ffmpeg

```powershell
# Windows (as Administrator)
choco install ffmpeg

# Verify installation
ffmpeg -version
```

### Problem: No input files found

**Solution:** Check input directory path

```powershell
# Verify files exist
ls ..\pre_inference\input_data_buffer\*.mp4

# If empty, copy some MP4 files there
```

### Problem: Models not found

**Solution:** Verify model paths in CONFIG

```python
# Check if files exist
import os
print(os.path.exists('../audio-classifier/speechbrain-classifier/fine_tuned_urbansound8k_8class.pth'))
print(os.path.exists('../image-classifier/YOLOv8/yolov8n.pt'))
```

## 📈 Sample Workflow

### Scenario: Monitor 1-hour of traffic footage

1. **Prepare:**

   - Place `traffic_footage.mp4` in input directory
   - Set `chunk_duration = 10.0` (creates 360 chunks for 1 hour)

2. **Configure:**

   ```python
   CONFIG = {
       'default_zone': NoiseZone.RESIDENTIAL,
       'base_timestamp': datetime(2025, 10, 30, 8, 0, 0),  # 8 AM
       'skip_preprocessing': False,
   }
   ```

3. **Run Pipeline:**

   - Execute all notebook cells
   - Wait for processing (~5-15 minutes depending on hardware)

4. **Analyze Results:**
   - Check compliance rate in summary
   - Identify peak violation times
   - Review horn and construction violations
   - Export CSV for stakeholder reporting

### Expected Processing Time

| Step                          | CPU Time       | GPU Time      |
| ----------------------------- | -------------- | ------------- |
| Pre-processing (1 hour video) | 2-3 min        | 2-3 min       |
| Audio Inference (360 chunks)  | 5-8 min        | 2-3 min       |
| Video Inference (360 chunks)  | 15-25 min      | 3-5 min       |
| Event Classification          | <1 min         | <1 min        |
| **Total**                     | **~25-35 min** | **~8-12 min** |

_Note: Times vary based on hardware and configuration_

## 🎯 Next Steps

After your first successful run:

1. **Explore Visualizations**: Review charts in notebook cells 6-10, 15
2. **Filter Events**: Use cell 13 to query specific violation types
3. **Compare Zones**: Run cell 14 to see how zone changes affect compliance
4. **Export Reports**: Cell 16 generates formatted compliance reports
5. **Customize Rules**: Edit `event_classifier.py` for custom regulations

## 💡 Pro Tips

1. **Incremental Processing**: Set `skip_preprocessing=True` to reprocess events with different zone settings without re-running heavy inference

2. **GPU Memory**: If using GPU, monitor memory with:

   ```python
   import torch
   print(torch.cuda.memory_summary())
   ```

3. **Batch Processing**: Process multiple videos by placing them all in input directory

4. **Time Stamping**: Set accurate `base_timestamp` for proper day/night classification

5. **Custom Zones**: Create custom zone limits by modifying `NOISE_LIMITS` in `event_classifier.py`

## 📚 Learn More

- **Full Documentation**: `INTEGRATED_PIPELINE_README.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Module Reference**: Check docstrings in `.py` files

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Verify all paths in CONFIG are correct
4. Ensure all dependencies are installed
5. Try with a smaller test video first

---

**Ready to start?** Open `inference.ipynb` and run the cells! 🚀
