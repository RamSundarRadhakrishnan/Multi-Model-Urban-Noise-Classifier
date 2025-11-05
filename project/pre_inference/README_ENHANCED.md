# Media Pre-Processing Pipeline - Enhanced with Chunking & Noise Analysis

Complete pipeline for processing MP4 videos with audio/video separation, chunking, and A-weighted noise level analysis.

## 🎯 Features

✅ **Audio/Video Separation** - Cleanly extracts both streams  
✅ **Configurable Chunking** - Splits into time segments (default: 10s)  
✅ **A-Weighted Noise Analysis** - Calculates SPL, Leq for each chunk  
✅ **Temporal Alignment** - Maintains sync through metadata  
✅ **Organized Storage** - Structured output directories  
✅ **Comprehensive Metadata** - JSON files with all analysis data  
✅ **CSV Export** - Easy data analysis in Excel/Pandas  
✅ **Visualization** - Plot noise levels over time

## 📂 Output Structure

```
processed_media/
├── audio/                    # Full audio files (WAV)
├── video/                    # Full video files (no audio)
├── audio_chunks/             # 10s audio segments
├── video_chunks/             # 10s video segments (synchronized)
├── metadata/                 # Original file metadata
├── chunk_metadata/           # Chunk info + noise levels
│   ├── master_chunk_index.json
│   └── [file]_chunks.json
└── noise_analysis/           # Noise statistics & CSV
    ├── [file]_noise_summary.json
    └── [file]_chunks.csv
```

## 🚀 Quick Start

```python
from integrated_processor import IntegratedMediaProcessor

# Initialize with 10-second chunks
processor = IntegratedMediaProcessor(
    input_dir='./input_data_buffer',
    chunk_duration=10.0
)

# Process all videos
results = processor.process_all()

# Get noise summary
summary = processor.get_noise_summary('video.mp4')
print(f"Mean noise: {summary['statistics']['a_weighted_spl']['mean']:.2f} dB(A)")
```

## 📊 Noise Metrics Calculated

For each audio chunk:

- **A-weighted SPL** - Sound Pressure Level in dB(A)
- **Leq** - Equivalent Continuous Sound Level in dB(A)
- **RMS** - Root Mean Square amplitude
- **Peak Level** - Maximum amplitude in dB
- **Crest Factor** - Peak-to-RMS ratio

## 📈 Example Output

### Chunk Metadata (JSON)

```json
{
  "chunk_index": 0,
  "start_time": 0.0,
  "end_time": 10.0,
  "audio_chunk": "video_chunk_0000.wav",
  "video_chunk": "video_chunk_0000.mp4",
  "noise_analysis": {
    "a_weighted_spl": 72.5,
    "leq_a": 71.8,
    "rms": 0.025,
    "peak_db": -12.3
  }
}
```

### Noise Statistics

```json
{
  "a_weighted_spl": {
    "mean": 70.2,
    "median": 69.8,
    "min": 55.3,
    "max": 85.6,
    "std": 8.4
  }
}
```

## 🔧 Configuration

Edit `config.json`:

```json
{
  "chunking": {
    "chunk_duration_seconds": 10.0
  },
  "noise_analysis": {
    "calculate_a_weighting": true,
    "calculate_leq": true,
    "export_csv": true
  }
}
```

## 📝 Dependencies

```bash
pip install numpy matplotlib
```

Plus **ffmpeg** (must be in PATH)

## 💡 Use Cases

1. **Urban Noise Monitoring** - Analyze noise patterns over time
2. **Audio Classification** - Pre-process data for ML models
3. **Video Segmentation** - Split long videos into manageable chunks
4. **Multi-Modal Analysis** - Synchronized audio/video for fusion models
5. **Compliance Monitoring** - Track noise levels against regulations

## 📚 Documentation

See `QUICKSTART.md` for immediate usage  
See full `README_DETAILED.md` for comprehensive documentation  
Run `examples.py` for code samples  
Open `pre_inference.ipynb` for interactive guide

## 🎬 Workflow

```
MP4 Video
    ↓
[1] Separate Audio/Video
    ↓
[2] Chunk into 10s segments
    ↓
[3] Analyze noise levels (A-weighted)
    ↓
[4] Generate metadata & CSV
    ↓
Ready for classification!
```

## 🔍 Key Classes

- `IntegratedMediaProcessor` - Main pipeline (all-in-one)
- `MediaProcessor` - Basic separation only
- `MediaChunker` - Chunking functionality
- `AudioAnalyzer` - Noise level calculations

## 📊 Integration

Outputs feed directly into:

- Audio classifier (`../audio-classifier/`)
- Image classifier (`../image-classifier/`)
- Post-inference fusion (`../post_inference/`)

Temporal alignment ensures synchronized multi-modal processing!
