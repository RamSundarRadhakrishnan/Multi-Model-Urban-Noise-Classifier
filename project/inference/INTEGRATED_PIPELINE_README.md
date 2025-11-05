# Integrated Urban Noise Classification & Event Detection System

A comprehensive pipeline for urban noise pollution monitoring and compliance analysis based on Indian Noise Pollution (Regulation and Control) Rules, 2000.

## Overview

This system integrates three major components:

1. **Media Pre-processing** - Chunk media files and analyze noise levels
2. **Multi-modal Inference** - Audio classification + Video object detection
3. **Event Classification** - Regulatory compliance analysis and violation detection

## Features

### 🎵 Audio Classification

- Fine-tuned SpeechBrain model for urban noise classification
- 8 noise categories: crowd noise, generator, horn, mobile music, community radio, construction, siren, car alarm
- Batch processing with probability vectors

### 🎬 Video Analysis

- Dual YOLO models (standard + MOCS construction equipment)
- Construction site detection
- Object detection and aggregation across frames

### 📊 Noise Analysis

- A-weighted sound pressure level (SPL) measurement
- Equivalent continuous sound level (Leq)
- Per-chunk noise statistics

### 🏛️ Regulatory Compliance

- Zone-based noise limits (Residential, Industrial, Silence, Commercial)
- Day/Night time period regulations
- Activity-specific restrictions (horns, loudspeakers, construction)
- Violation severity classification

## Noise Zone Regulations

| Zone            | Day Limit (6am-10pm) | Night Limit (10pm-6am) |
| --------------- | -------------------- | ---------------------- |
| **Residential** | ≤ 55 dB(A)           | ≤ 45 dB(A)             |
| **Industrial**  | ≤ 75 dB(A)           | ≤ 70 dB(A)             |
| **Silence**     | ≤ 50 dB(A)           | ≤ 40 dB(A)             |
| **Commercial**  | ≤ 65 dB(A)           | ≤ 55 dB(A)             |

### Activity Restrictions

| Activity                 | Residential              | Industrial | Silence       |
| ------------------------ | ------------------------ | ---------- | ------------- |
| **Horn usage (night)**   | ❌ Prohibited            | ✅ Allowed | ❌ Prohibited |
| **Loudspeakers (night)** | ❌ Prohibited            | Restricted | ❌ Prohibited |
| **Construction (night)** | ❌ Prohibited (10pm-6am) | ✅ Allowed | ❌ Prohibited |

## Installation

### Prerequisites

```bash
# System requirements
- Python 3.7+
- ffmpeg (for media processing)
- CUDA-capable GPU (optional, for faster inference)
```

### Install Dependencies

```bash
cd project/inference
pip install -r requirements.txt
```

### Required Models

1. **SpeechBrain Audio Model**

   - Pretrained model: `pretrained_urbansound8k`
   - Fine-tuned weights: `fine_tuned_urbansound8k_8class.pth`

2. **YOLO Video Models**
   - Standard YOLOv8: `yolov8n.pt`
   - MOCS fine-tuned: `mocs_yolov8n_train/weights/best.pt`

## Project Structure

```
inference/
├── inference.ipynb              # Main integrated pipeline notebook
├── integrated_inference.py      # Pipeline orchestration module
├── event_classifier.py          # Regulatory compliance classifier
├── event_storage.py            # Event storage and reporting
├── requirements.txt            # Python dependencies
└── output/                     # Generated outputs
    └── events/
        ├── json/               # Event data in JSON format
        ├── csv/                # Event data in CSV format
        └── summaries/          # Summary reports
```

## Usage

### Quick Start

1. **Prepare Input Data**

   ```bash
   # Place MP4 files in input directory
   mkdir -p ../pre_inference/input_data_buffer
   # Copy your media files there
   ```

2. **Run the Integrated Notebook**

   - Open `inference.ipynb` in Jupyter/VS Code
   - Configure settings in **Section 2: Configuration**
   - Run all cells to execute the complete pipeline

3. **Review Results**
   - Check `output/events/` directory for results
   - View visualizations and statistics in notebook
   - Read compliance reports

### Configuration

Edit the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    # Directories
    'input_dir': '../pre_inference/input_data_buffer',
    'event_output_dir': './output/events',

    # Processing parameters
    'chunk_duration': 10.0,  # seconds
    'default_zone': NoiseZone.RESIDENTIAL,

    # Model paths and settings
    'audio_pretrained_model': '...',
    'yolo_standard_model': '...',

    # Skip steps (use existing data)
    'skip_preprocessing': False,
    'skip_audio_inference': False,
    'skip_video_inference': False,
}
```

### Using Individual Modules

#### Event Classifier

```python
from event_classifier import EventClassifier, NoiseZone
from datetime import datetime

classifier = EventClassifier(zone=NoiseZone.RESIDENTIAL)

event = classifier.classify_event(
    event_id="chunk_001",
    timestamp=datetime.now(),
    noise_level_db=60.5,
    audio_classes={"motorvehicle-horn": 0.85},
    video_classes={"car": 0.92},
    is_construction_site=False,
    construction_confidence=0.0
)

print(f"Compliant: {event.is_compliant}")
print(f"Violations: {event.violations}")
print(f"Recommendations: {event.recommendations}")
```

#### Event Storage

```python
from event_storage import EventStorage

storage = EventStorage(storage_dir="./output/events")
storage.add_event(event)

# Save in multiple formats
storage.save_json()
storage.save_csv()
storage.save_violations_only()
storage.generate_summary_report()

# Filter and analyze
violations = storage.get_violations_only()
severe = storage.filter_by_severity('severe')
```

#### Integrated Pipeline

```python
from integrated_inference import IntegratedInferencePipeline
from event_classifier import NoiseZone

pipeline = IntegratedInferencePipeline(
    input_dir="./input",
    preprocessing_output_dir="./processed",
    audio_pretrained_model="path/to/model",
    yolo_standard_model="path/to/yolo",
    yolo_mocs_model="path/to/mocs",
    default_zone=NoiseZone.RESIDENTIAL
)

events, storage = pipeline.run_full_pipeline(
    zone=NoiseZone.RESIDENTIAL,
    save_results=True
)
```

## Output Formats

### Event JSON

```json
{
  "event_id": "video_chunk_0001",
  "zone": "residential",
  "time_of_day": "night",
  "timestamp": "2025-10-30T22:30:00",
  "noise_level_db": 58.3,
  "ambient_limit_db": 45.0,
  "excess_db": 13.3,
  "violations": ["ambient_exceeded", "horn_prohibited"],
  "severity": "moderate",
  "is_compliant": false,
  "recommendations": [
    "Reduce ambient noise by 13.3 dB...",
    "Horn usage is prohibited..."
  ]
}
```

### CSV Export

- `events_TIMESTAMP.csv` - Summary view
- `events_detailed_TIMESTAMP.csv` - All audio/video classes
- `violations_TIMESTAMP.json` - Violations only

### Summary Report

- Total events and compliance rate
- Violation type breakdown
- Zone and time analysis
- Noise statistics
- Construction activity metrics

## Violation Detection

The system detects the following violation types:

1. **Ambient Noise Exceeded** - Noise level exceeds zone limit
2. **Horn Prohibited** - Horn usage in restricted zones/times
3. **Loudspeaker Prohibited** - PA system use during prohibited hours
4. **Construction Prohibited** - Construction activity during restricted times
5. **Multiple Violations** - Combination of 2+ violations

### Severity Levels

- **Compliant** - No violations
- **Minor** - 1 violation OR 5-10 dB excess
- **Moderate** - 2 violations OR 10-15 dB excess
- **Severe** - 3+ violations OR >15 dB excess

## Customization

### Add New Audio Classes

Edit `audio_class_names` in config and retrain the audio model.

### Adjust Noise Limits

Modify `NOISE_LIMITS` in `event_classifier.py`:

```python
NOISE_LIMITS = {
    NoiseZone.RESIDENTIAL: {
        TimeOfDay.DAY: 55.0,
        TimeOfDay.NIGHT: 45.0
    },
    # Add custom zones...
}
```

### Change Time Periods

Edit `get_time_of_day()` method in `EventClassifier`:

```python
def get_time_of_day(self, timestamp: datetime) -> TimeOfDay:
    hour = timestamp.hour
    if 6 <= hour < 22:  # Customize these hours
        return TimeOfDay.DAY
    else:
        return TimeOfDay.NIGHT
```

### Custom Violation Rules

Add new violation checks in `EventClassifier.classify_event()`.

## Performance Tips

1. **GPU Acceleration**: Set `device='cuda'` for faster YOLO inference
2. **Frame Skip**: Increase `frame_skip` to process fewer frames (faster, less accurate)
3. **Batch Size**: Adjust `audio_batch_size` based on GPU memory
4. **Skip Steps**: Use existing preprocessed data with `skip_preprocessing=True`

## Troubleshooting

### Import Errors

```python
# Add parent directories to Python path
sys.path.append(str(Path.cwd().parent / "pre_inference"))
sys.path.append(str(Path.cwd().parent / "audio-classifier/speechbrain-classifier"))
```

### CUDA Out of Memory

```python
# Reduce batch sizes
'audio_batch_size': 4,
'frame_skip': 60,  # Process fewer frames
```

### FFmpeg Not Found

```bash
# Windows (PowerShell as admin)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## API Reference

### EventClassifier

**Methods:**

- `classify_event()` - Classify a single event
- `get_noise_limit()` - Get limit for zone/time
- `check_ambient_noise_violation()` - Check noise level
- `check_horn_violation()` - Check horn usage
- `check_loudspeaker_violation()` - Check PA system
- `check_construction_violation()` - Check construction activity

### EventStorage

**Methods:**

- `add_event()` - Add single event
- `save_json()` - Export to JSON
- `save_csv()` - Export to CSV
- `generate_summary_report()` - Create summary
- `filter_by_zone()` - Filter by noise zone
- `filter_by_severity()` - Filter by severity
- `get_violations_only()` - Get non-compliant events

### IntegratedInferencePipeline

**Methods:**

- `run_full_pipeline()` - Execute complete pipeline
- `run_preprocessing()` - Media chunking and noise analysis
- `run_audio_inference()` - Audio classification
- `run_video_inference()` - Video object detection
- `classify_events()` - Regulatory compliance analysis

## Citation

If you use this system in your research, please cite:

```
Urban Noise Classification and Event Detection System
Based on Indian Noise Pollution (Regulation and Control) Rules, 2000
```

## License

[Include your license information here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## Contact

[Your contact information]

---

**Last Updated:** October 30, 2025
