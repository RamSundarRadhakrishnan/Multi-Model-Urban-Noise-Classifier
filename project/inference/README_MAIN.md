# 🎯 Integrated Urban Noise Classification Pipeline

> **Complete end-to-end system for urban noise monitoring and regulatory compliance analysis**

[![Status](https://img.shields.io/badge/status-ready-green)]()
[![Python](https://img.shields.io/badge/python-3.7+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place MP4 files in input directory
mkdir ../pre_inference/input_data_buffer
# Copy your videos there

# 3. Run the notebook
jupyter notebook inference.ipynb
# Execute all cells → Get compliance reports!
```

**⏱️ Expected Time:** ~10-30 minutes for 1 hour of video (depending on hardware)

---

## 📖 What This System Does

### Input

- 🎥 MP4 video files (with audio)

### Output

- ✅ **Event Classifications** - Each 10-second chunk analyzed
- 📊 **Compliance Reports** - Zone-based regulatory analysis
- 🚨 **Violation Detection** - Identify non-compliant activities
- 📈 **Statistical Analysis** - Noise trends and patterns
- 💾 **Multi-Format Export** - JSON, CSV, summary reports

---

## 🏛️ Based on Indian Noise Pollution Rules

| Zone            | Day Limit | Night Limit |
| --------------- | --------- | ----------- |
| **Residential** | 55 dB(A)  | 45 dB(A)    |
| **Industrial**  | 75 dB(A)  | 70 dB(A)    |
| **Silence**     | 50 dB(A)  | 40 dB(A)    |
| **Commercial**  | 65 dB(A)  | 55 dB(A)    |

**Detects violations:** Ambient noise exceeded, horn usage, loudspeakers, construction activity

---

## 📁 Key Files

| File                            | Purpose                              | Start Here    |
| ------------------------------- | ------------------------------------ | ------------- |
| `inference.ipynb`               | **Main notebook** - Run the pipeline | ⭐ **START**  |
| `QUICKSTART_INTEGRATED.md`      | 5-minute setup guide                 | 📖 Read first |
| `INTEGRATED_PIPELINE_README.md` | Complete documentation               | 📚 Reference  |
| `SYSTEM_OVERVIEW.md`            | Architecture & design                | 🏗️ Understand |
| `CLASSIFICATION_LOGIC.md`       | How violations are detected          | 🎓 Learn      |

### Python Modules

- `integrated_inference.py` - Pipeline orchestrator
- `event_classifier.py` - Regulatory compliance logic
- `event_storage.py` - Data persistence & reporting

---

## 🎨 Features

### ✨ Multi-Modal Analysis

- 🎵 **Audio Classification** - 8 noise categories (horns, sirens, construction, etc.)
- 🎬 **Video Detection** - Objects & construction sites via dual YOLO
- 📊 **Noise Measurement** - A-weighted SPL and Leq calculations

### 🏷️ Smart Classification

- ⏰ **Time-Aware** - Day/night period detection
- 🗺️ **Zone-Based** - Different limits for different areas
- 🎯 **Activity Detection** - Horns, loudspeakers, construction equipment
- 📈 **Severity Levels** - Compliant, minor, moderate, severe

### 📤 Comprehensive Output

- `events_TIMESTAMP.json` - All event data
- `violations_TIMESTAMP.json` - Non-compliant events only
- `events_TIMESTAMP.csv` - Summary spreadsheet
- `summary_TIMESTAMP.json` - Statistical overview
- `compliance_report_TIMESTAMP.txt` - Formatted report

---

## 💡 Example Use Cases

### 🏘️ Residential Area Monitoring

Monitor noise levels near apartments, detect horn violations at night, track construction timing compliance.

### 🏗️ Construction Site Compliance

Ensure construction activities follow time restrictions, measure noise impact on nearby areas.

### 🏥 Hospital Zone Enforcement

Strict monitoring of silence zones, detect any violations immediately, generate evidence for enforcement.

### 📊 Urban Planning Analysis

Analyze noise patterns over time, identify hotspots, inform policy decisions with data.

---

## 📊 Sample Output

```json
{
  "event_id": "traffic_chunk_0042",
  "zone": "residential",
  "time_of_day": "night",
  "noise_level_db": 62.3,
  "ambient_limit_db": 45.0,
  "excess_db": 17.3,
  "violations": ["ambient_exceeded", "horn_prohibited"],
  "severity": "severe",
  "is_compliant": false,
  "recommendations": [
    "Reduce ambient noise by 17.3 dB to comply with residential zone nighttime limits",
    "Horn usage is prohibited in residential zones during nighttime"
  ]
}
```

---

## 🛠️ Configuration

Edit the `CONFIG` dictionary in `inference.ipynb`:

```python
CONFIG = {
    # Set your zone type
    'default_zone': NoiseZone.RESIDENTIAL,  # or INDUSTRIAL, SILENCE, COMMERCIAL

    # Processing parameters
    'chunk_duration': 10.0,  # seconds per chunk
    'frame_skip': 30,  # process every 30th frame

    # Model paths
    'audio_pretrained_model': '../audio-classifier/speechbrain-classifier/pretrained_urbansound8k',
    'yolo_standard_model': '../image-classifier/YOLOv8/yolov8n.pt',

    # Skip steps to reprocess faster
    'skip_preprocessing': False,
    'skip_audio_inference': False,
    'skip_video_inference': False,
}
```

---

## 📚 Documentation Guide

**New to the system?**

1. 📖 Read `QUICKSTART_INTEGRATED.md` (5 minutes)
2. ▶️ Run `inference.ipynb` with sample data
3. 📊 Review generated reports and visualizations

**Want to understand how it works?**

1. 🏗️ Read `SYSTEM_OVERVIEW.md` for architecture
2. 🎓 Read `CLASSIFICATION_LOGIC.md` for violation rules
3. 💻 Explore module docstrings in Python files

**Need detailed reference?**

1. 📚 Check `INTEGRATED_PIPELINE_README.md`
2. 🔍 Review API documentation in code
3. 💬 See FAQ and troubleshooting sections

---

## 🎯 Architecture

```
┌─────────────────────────────────────────────────┐
│              Input: MP4 Videos                   │
└──────────────────┬──────────────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │   Pre-processing Module     │
    │  (chunk + noise analysis)   │
    └──────────────┬──────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼──────┐    ┌────────▼────────┐
│    Audio     │    │      Video      │
│  Inference   │    │    Inference    │
│ (SpeechBrain)│    │   (Dual YOLO)   │
└───────┬──────┘    └────────┬────────┘
        │                     │
        └──────────┬──────────┘
                   │
          ┌────────▼────────┐
          │  Data Matching  │
          │   & Alignment   │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │     Event       │
          │ Classification  │
          │  (Regulations)  │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │     Storage     │
          │   & Reporting   │
          └─────────────────┘
                   │
    ┌──────────────┴──────────────┐
    │   Output: Compliance Data   │
    └─────────────────────────────┘
```

---

## 🔧 System Requirements

### Minimum

- Python 3.7+
- 4GB RAM
- CPU: Any modern processor
- Storage: 10GB free space

### Recommended

- Python 3.9+
- 16GB RAM
- GPU: NVIDIA CUDA-capable (for faster processing)
- Storage: 50GB+ for large datasets

### Software Dependencies

- ffmpeg (for media processing)
- CUDA Toolkit (optional, for GPU acceleration)

---

## 📈 Performance

| Processing Step               | CPU Time       | GPU Time      |
| ----------------------------- | -------------- | ------------- |
| Pre-processing (1 hour video) | 2-3 min        | 2-3 min       |
| Audio Inference (360 chunks)  | 5-8 min        | 2-3 min       |
| Video Inference (360 chunks)  | 15-25 min      | 3-5 min       |
| Event Classification          | <1 min         | <1 min        |
| **Total Pipeline**            | **~25-35 min** | **~8-12 min** |

_Times for 1 hour of 1080p video at 30fps_

---

## 🆘 Troubleshooting

### "Module not found" errors

```bash
# Ensure you're in the correct directory
cd d:\Projects\UrbanNoiseClassifier\project\inference

# Verify Python paths in notebook
```

### CUDA out of memory

```python
# Reduce batch sizes in CONFIG
'audio_batch_size': 4,
'frame_skip': 60,
'device': 'cpu'  # Use CPU instead
```

### No videos found

```bash
# Check input directory
ls ../pre_inference/input_data_buffer/*.mp4

# Copy videos there
cp /path/to/videos/*.mp4 ../pre_inference/input_data_buffer/
```

**More help:** See troubleshooting section in `QUICKSTART_INTEGRATED.md`

---

## 🎓 Learning Path

### Beginner

1. ✅ Install dependencies
2. ✅ Run notebook with sample video
3. ✅ Review visualizations
4. ✅ Understand compliance reports

### Intermediate

1. ✅ Modify CONFIG for different zones
2. ✅ Filter events by severity
3. ✅ Export custom CSV reports
4. ✅ Compare different time periods

### Advanced

1. ✅ Customize violation rules
2. ✅ Add new audio/video classes
3. ✅ Implement custom aggregation
4. ✅ Extend storage formats

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Real-time processing
- Web dashboard
- Mobile app integration
- Database backend
- Additional noise zones
- Multi-language support

---

## 📄 License

[Include your license here]

---

## 📧 Contact

[Your contact information]

---

## 🎉 Get Started Now!

```bash
# Quick start command
jupyter notebook inference.ipynb
```

Then run all cells and see your compliance reports! 🚀

---

**Documentation Tree:**

```
📦 inference/
├── 📘 README.md (this file)           ← Overview & quick start
├── 📗 QUICKSTART_INTEGRATED.md        ← 5-minute setup guide
├── 📕 INTEGRATED_PIPELINE_README.md   ← Complete reference
├── 📙 SYSTEM_OVERVIEW.md              ← Architecture & design
├── 📔 CLASSIFICATION_LOGIC.md         ← Violation rules explained
├── 📄 FILES_CREATED.md                ← What was built
├── 📓 inference.ipynb                 ← **RUN THIS!**
├── 🐍 integrated_inference.py         ← Pipeline code
├── 🐍 event_classifier.py             ← Classification logic
├── 🐍 event_storage.py                ← Data management
└── 📋 requirements.txt                ← Dependencies
```

**→ Start with `QUICKSTART_INTEGRATED.md` or jump directly to `inference.ipynb`** ✨

---

_Last updated: October 30, 2025_
