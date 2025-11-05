# Pre-Processing Pipeline - Complete Feature Summary

## 🎯 What's New

Your media pre-processing pipeline now includes:

### ✨ New Features Added

1. **Configurable Chunking**

   - Split videos into time segments (default: 10 seconds)
   - Synchronized audio and video chunks
   - Maintains temporal alignment

2. **A-Weighted Noise Level Analysis**

   - Sound Pressure Level (SPL) in dB(A) for each chunk
   - Equivalent Continuous Level (Leq) in dB(A)
   - RMS, Peak, and Crest Factor calculations
   - Statistical summaries (mean, median, min, max, std)

3. **Enhanced Metadata**

   - Chunk-level metadata with noise data
   - Master index of all chunks
   - CSV export for easy analysis

4. **Visualization**
   - Plot noise levels over time
   - Identify high/low noise segments
   - Temporal pattern analysis

---

## 📦 Complete Module List

### Core Modules

1. **`media_processor.py`** (Original)

   - Basic audio/video separation
   - Full file extraction

2. **`media_chunker.py`** (NEW)

   - Splits audio/video into chunks
   - Configurable chunk duration
   - Synchronized chunking

3. **`audio_analyzer.py`** (NEW)

   - A-weighting filter implementation
   - SPL and Leq calculations
   - Comprehensive audio metrics
   - Batch analysis support

4. **`integrated_processor.py`** (NEW)

   - Complete pipeline integration
   - Separation + Chunking + Analysis
   - All-in-one processing

5. **`media_utils.py`** (Enhanced)
   - Dependency checking
   - Summary generation
   - Helper functions

### Interfaces

6. **`pre_inference.ipynb`** (Enhanced)

   - Interactive notebook with 10 sections
   - Visualization of noise levels
   - CSV export
   - Chunk inspection

7. **`process_videos.py`** (Original)

   - Command-line interface
   - Batch processing

8. **`examples.py`** (Original)
   - Usage demonstrations

### Configuration

9. **`config.json`** (Enhanced)

   - Chunking parameters
   - Noise analysis settings
   - Output structure configuration

10. **`requirements.txt`** (NEW)
    - Python dependencies (numpy, matplotlib)

### Documentation

11. **`README.md`** - Original comprehensive guide
12. **`README_ENHANCED.md`** (NEW) - Quick reference for new features
13. **`QUICKSTART.md`** (Enhanced) - Updated quick start
14. **`FEATURE_SUMMARY.md`** - This file

---

## 🔄 Processing Workflow

```
Input: MP4 Video File
    ↓
┌─────────────────────────────────────┐
│ Step 1: Audio/Video Separation      │
│ - Extract audio → WAV               │
│ - Extract video → MP4 (no audio)    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 2: Chunking                    │
│ - Split audio into 10s segments     │
│ - Split video into 10s segments     │
│ - Maintain synchronization          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 3: Noise Analysis              │
│ - Calculate A-weighted SPL          │
│ - Calculate Leq                     │
│ - Compute RMS, Peak, Crest Factor   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step 4: Metadata Generation         │
│ - Create chunk metadata (JSON)      │
│ - Generate noise summaries          │
│ - Export to CSV                     │
└─────────────────────────────────────┘
    ↓
Output: Organized chunks + metadata + analysis
```

---

## 📊 Output Files Generated

For each input video `example.mp4`:

### Full Files

- `audio/example.wav` - Complete audio
- `video/example_video_only.mp4` - Complete video

### Chunks (10-second segments)

- `audio_chunks/example_chunk_0000.wav` → `example_chunk_0001.wav` → ...
- `video_chunks/example_chunk_0000.mp4` → `example_chunk_0001.mp4` → ...

### Metadata

- `metadata/example_metadata.json` - Original file info
- `chunk_metadata/example_chunks.json` - All chunks with noise data
- `chunk_metadata/master_chunk_index.json` - Index of all files

### Noise Analysis

- `noise_analysis/example_noise_summary.json` - Statistics
- `noise_analysis/example_chunks.csv` - Spreadsheet format

---

## 📈 Noise Metrics Explained

### A-Weighted SPL (dB(A))

- Sound Pressure Level adjusted for human hearing sensitivity
- Mimics how humans perceive loudness
- Higher values = louder perceived sound

### Leq (dB(A))

- Equivalent Continuous Sound Level
- Average noise level over the chunk duration
- Standard metric for environmental noise assessment

### RMS (Root Mean Square)

- Average amplitude of the audio signal
- Linear scale (0.0 to 1.0)

### Peak (dB)

- Maximum amplitude in the chunk
- Useful for detecting transient loud events

### Crest Factor (dB)

- Ratio of peak to RMS
- Indicates signal dynamics
- High values = impulsive sounds (e.g., gunshots)
- Low values = steady sounds (e.g., traffic hum)

---

## 💡 Use Cases

### 1. Urban Noise Monitoring

- Track noise levels across different times of day
- Identify noise pollution hotspots
- Generate compliance reports

### 2. Audio Event Detection

- Pre-process data for ML classifiers
- Chunk long recordings into manageable segments
- Filter chunks by noise level

### 3. Multi-Modal Classification

- Synchronized audio/video for fusion models
- Temporal alignment ensures consistency
- Feed into your urban noise classifier

### 4. Data Quality Assessment

- Identify silent or corrupt segments
- Filter low-quality recordings
- Validate recording equipment

### 5. Research & Analysis

- Export to CSV for statistical analysis
- Visualize temporal patterns
- Compare noise profiles across locations

---

## 🔧 Customization Options

### Chunk Duration

```python
# 5-second chunks
processor = IntegratedMediaProcessor(
    input_dir='./input_data_buffer',
    chunk_duration=5.0
)

# 30-second chunks
processor = IntegratedMediaProcessor(
    input_dir='./input_data_buffer',
    chunk_duration=30.0
)
```

### Audio Format

```python
# MP3 instead of WAV
processor.process_all(audio_format='mp3')
```

### Selective Processing

```python
# Process only specific file
processor.process_file_complete('specific_video.mp4')

# Process files matching pattern
processor.process_all(file_pattern='scene*.mp4')
```

---

## 📚 API Quick Reference

### IntegratedMediaProcessor

```python
# Initialize
processor = IntegratedMediaProcessor(input_dir, output_base_dir, chunk_duration)

# Process all files
results = processor.process_all(audio_format='wav')

# Process single file
result = processor.process_file_complete('video.mp4')

# Get noise summary
summary = processor.get_noise_summary('video.mp4')

# Get chunk info
chunk = processor.get_chunk_info('video.mp4', chunk_index=0)
```

### AudioAnalyzer

```python
analyzer = AudioAnalyzer()

# Analyze single audio file
metrics = analyzer.analyze_audio_clip(audio_path)

# Batch analysis
results = analyzer.analyze_audio_clips_batch(audio_paths)
```

### MediaChunker

```python
chunker = MediaChunker(chunk_duration=10.0)

# Chunk audio
audio_chunks = chunker.chunk_audio(input_path, output_dir)

# Chunk video
video_chunks = chunker.chunk_video(input_path, output_dir)

# Chunk both (synchronized)
audio_chunks, video_chunks = chunker.chunk_media_pair(
    audio_path, video_path, audio_out_dir, video_out_dir
)
```

---

## ✅ Checklist for First Use

- [ ] Install ffmpeg and add to PATH
- [ ] Install Python dependencies: `pip install numpy matplotlib`
- [ ] Place MP4 files in `input_data_buffer/`
- [ ] Run notebook: Open `pre_inference.ipynb`
- [ ] Execute cells sequentially
- [ ] Check outputs in `processed_media/`
- [ ] Review noise analysis summaries
- [ ] Export CSV for further analysis
- [ ] Visualize noise levels

---

## 🎓 Learning Path

1. **Start Here**: `QUICKSTART.md`
2. **Interactive Demo**: `pre_inference.ipynb`
3. **Code Examples**: `examples.py`
4. **Full Documentation**: `README.md`
5. **Feature Overview**: This file

---

## 🚀 Next Steps

Your processed chunks are now ready for:

1. **Audio Classification** → Feed chunks to your audio classifier
2. **Image Classification** → Feed video frames to image classifier
3. **Post-Inference Fusion** → Combine predictions with noise levels
4. **Noise Level Integration** → Use noise data as additional features

The temporal alignment metadata ensures everything stays synchronized across your multi-modal pipeline!

---

## 📞 Support

- Check notebook for examples
- Review `examples.py` for code patterns
- See `README_ENHANCED.md` for quick reference
- Full API docs in `README.md`
