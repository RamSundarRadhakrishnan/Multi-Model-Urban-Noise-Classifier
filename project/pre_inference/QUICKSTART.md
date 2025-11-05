# Quick Start Guide - Media Pre-Processing

## TL;DR

1. **Install ffmpeg** (if not already installed)
2. **Place MP4 files** in `input_data_buffer/` folder
3. **Run the notebook** `pre_inference.ipynb` or command-line script
4. **Find outputs** in `processed_media/` folder with audio, video, and metadata

---

## What This Does

Processes MP4 video files by:

- ✅ Extracting audio as WAV files
- ✅ Extracting video without audio
- ✅ Creating JSON metadata to maintain temporal alignment
- ✅ Organizing everything in a structured output directory

---

## Quick Setup

### 1. Install ffmpeg

**Windows:**

```powershell
# Download from https://ffmpeg.org/download.html
# Add to PATH environment variable
```

**Check installation:**

```powershell
ffmpeg -version
```

### 2. Add Your Videos

Place your MP4 files in:

```
project/pre_inference/input_data_buffer/
```

### 3. Run Processing

**Option A - Notebook (Recommended):**
Open and run `pre_inference.ipynb`

**Option B - Command Line:**

```powershell
cd project/pre_inference
python process_videos.py
```

---

## File Structure After Processing

```
pre_inference/
├── input_data_buffer/
│   └── my_video.mp4                    # Your input file
│
└── processed_media/
    ├── audio/
    │   └── my_video.wav                # Full audio
    ├── video/
    │   └── my_video_video_only.mp4     # Full video (no audio)
    ├── audio_chunks/
    │   ├── my_video_chunk_0000.wav     # 0-10s audio
    │   ├── my_video_chunk_0001.wav     # 10-20s audio
    │   └── ...
    ├── video_chunks/
    │   ├── my_video_chunk_0000.mp4     # 0-10s video
    │   ├── my_video_chunk_0001.mp4     # 10-20s video
    │   └── ...
    ├── chunk_metadata/
    │   ├── my_video_chunks.json        # Chunk info + noise levels
    │   └── master_chunk_index.json     # All files index
    ├── noise_analysis/
    │   ├── my_video_noise_summary.json # Noise statistics
    │   └── my_video_chunks.csv         # CSV export
    └── metadata/
        └── my_video_metadata.json      # Original file metadata
```

---

## Understanding the Output

### Audio Files (`audio/`)

- Format: WAV (or MP3/AAC if configured)
- Sample Rate: 44100 Hz
- Channels: Stereo

### Video Files (`video/`)

- Format: MP4 (video stream only)
- Original codec preserved
- No audio track

### Metadata Files (`metadata/`)

Each video gets a JSON file with:

```json
{
  "temporal_alignment": {
    "start_time": 0.0,
    "duration": 120.5,
    "frame_rate": "30/1",
    "sample_rate": "44100"
  }
}
```

This ensures audio and video stay synchronized!

---

## Common Commands

### Check if ffmpeg is installed

```powershell
python process_videos.py --check-deps
```

### Process all MP4 files

```powershell
python process_videos.py
```

### Process with MP3 audio instead of WAV

```powershell
python process_videos.py -f mp3
```

### Custom input/output directories

```powershell
python process_videos.py -i ./my_videos -o ./my_output
```

### View summary of processed files

```powershell
python process_videos.py --summary
```

---

## Module Files

| File                  | Purpose                |
| --------------------- | ---------------------- |
| `media_processor.py`  | Main processing class  |
| `media_utils.py`      | Helper functions       |
| `pre_inference.ipynb` | Interactive notebook   |
| `process_videos.py`   | Command-line interface |
| `examples.py`         | Usage examples         |
| `config.json`         | Configuration file     |
| `README.md`           | Full documentation     |
| `QUICKSTART.md`       | This file              |

---

## Troubleshooting

### "ffmpeg not found"

- Install ffmpeg from https://ffmpeg.org/download.html
- Add to system PATH
- Restart terminal/VS Code

### "No files found"

- Check MP4 files are in `input_data_buffer/`
- Verify file extension is `.mp4`
- Check file permissions

### Processing fails

- Verify file isn't corrupted: `ffmpeg -i your_file.mp4`
- Ensure sufficient disk space
- Check ffmpeg can read the codec

---

## Next Steps

After processing, use the separated audio and video for:

1. **Audio Classification** → `../audio-classifier/`
2. **Image/Video Classification** → `../image-classifier/`
3. **Multi-Modal Fusion** → `../post_inference/`

The metadata ensures everything stays temporally aligned! 🎯

---

## Need Help?

- See `README.md` for detailed documentation
- Run `examples.py` to see usage examples
- Check the notebook for interactive walkthrough
