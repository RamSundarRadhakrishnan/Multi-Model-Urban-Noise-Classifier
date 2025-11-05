# Media Pre-Processing Pipeline

This module processes MP4 video files by separating audio and video streams while maintaining temporal alignment through metadata mapping.

## Overview

The pre-processing pipeline performs the following tasks:

1. **Separate audio and video** - Extracts audio and video streams from MP4 files
2. **Organize outputs** - Stores audio and video in separate directories
3. **Maintain temporal alignment** - Creates metadata JSON files that map audio and video clips with timestamps, duration, and other synchronization information

## Directory Structure

```
pre_inference/
├── media_processor.py       # Basic audio/video separation
├── media_chunker.py         # Chunking module
├── audio_analyzer.py        # A-weighted noise level analysis
├── integrated_processor.py  # Integrated pipeline (separation + chunking + analysis)
├── media_utils.py           # Utility functions
├── pre_inference.ipynb      # Interactive notebook
├── process_videos.py        # Command-line interface
├── examples.py              # Usage examples
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── QUICKSTART.md            # Quick start guide
├── input_data_buffer/       # Place MP4 files here
└── processed_media/         # Output directory (created automatically)
    ├── audio/               # Full extracted audio files (WAV)
    ├── video/               # Full video files without audio (MP4)
    ├── audio_chunks/        # Audio chunks (10s segments)
    ├── video_chunks/        # Video chunks (10s segments)
    ├── metadata/            # Original file metadata
    ├── chunk_metadata/      # Chunk-level metadata with noise data
    │   ├── master_chunk_index.json     # Index of all chunks
    │   └── [filename]_chunks.json      # Per-file chunk metadata
    └── noise_analysis/      # Noise analysis results
        └── [filename]_noise_summary.json  # Noise statistics
        └── [filename]_chunks.csv          # CSV export
```

## Requirements

### Software Dependencies

- **Python 3.7+**
- **ffmpeg** - Must be installed and accessible in PATH
  - Download: https://ffmpeg.org/download.html
  - Windows: Download binary and add to PATH
  - Linux: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

### Python Packages

No additional Python packages required beyond standard library.

## Installation

1. Ensure ffmpeg is installed:

   ```bash
   ffmpeg -version
   ```

2. Place your MP4 files in the `input_data_buffer/` directory

## Usage

### Option 1: Using the Notebook (Recommended)

1. Open `pre_inference.ipynb` in Jupyter or VS Code
2. Run cells sequentially to:
   - Check dependencies
   - Configure directories
   - Process all MP4 files
   - View results and metadata

### Option 2: Using Python Script

```python
from integrated_processor import IntegratedMediaProcessor

# Initialize processor with chunking
processor = IntegratedMediaProcessor(
    input_dir='./input_data_buffer',
    output_base_dir='./processed_media',
    chunk_duration=10.0  # 10 second chunks
)

# Process all MP4 files (separation + chunking + noise analysis)
results = processor.process_all(audio_format='wav')

# Get noise summary for a file
noise_summary = processor.get_noise_summary('your_video.mp4')
print(f"Mean SPL: {noise_summary['statistics']['a_weighted_spl']['mean']:.2f} dB(A)")

# Get info for a specific chunk
chunk_info = processor.get_chunk_info('your_video.mp4', chunk_index=0)
print(f"Chunk 0 SPL: {chunk_info['noise_analysis']['a_weighted_spl']:.2f} dB(A)")
```

### Option 3: Command Line Script

Create a simple script `process_videos.py`:

```python
from pathlib import Path
from media_processor import MediaProcessor
from media_utils import check_dependencies, print_dependency_status

# Check dependencies
deps = check_dependencies()
print_dependency_status(deps)

if not all(deps.values()):
    print("Error: Missing dependencies!")
    exit(1)

# Process videos
processor = MediaProcessor('./input_data_buffer')
results = processor.process_all(audio_format='wav')

print(f"Processed {len(results)} files successfully!")
```

Run it:

```bash
python process_videos.py
```

## Output Format

### Audio Files

- **Format**: WAV (configurable to MP3, AAC, etc.)
- **Sample Rate**: 44100 Hz
- **Channels**: Stereo (2 channels)
- **Naming**: `[original_filename].wav`

### Video Files

- **Format**: MP4 (video stream only, no audio)
- **Codec**: Original codec preserved (copy)
- **Naming**: `[original_filename]_video_only.mp4`

### Metadata JSON

Each processed file gets a metadata JSON file containing:

```json
{
  "original_file": "video.mp4",
  "audio_file": "video.wav",
  "video_file": "video_video_only.mp4",
  "processed_timestamp": "2025-10-29T12:00:00",
  "temporal_alignment": {
    "start_time": 0.0,
    "duration": 120.5,
    "frame_rate": "30/1",
    "sample_rate": "44100"
  },
  "video_codec": "h264",
  "video_width": 1920,
  "video_height": 1080,
  "audio_codec": "aac",
  "audio_channels": 2
}
```

### Master Index

`master_index.json` contains a summary of all processed files:

```json
{
  "processed_date": "2025-10-29T12:00:00",
  "total_files": 5,
  "successful": 5,
  "failed": 0,
  "files": {
    "video1.mp4": {
      /* metadata */
    },
    "video2.mp4": {
      /* metadata */
    }
  }
}
```

### Aligned Pairs Mapping

`aligned_pairs_mapping.json` provides a simple reference:

```json
{
  "pairs": [
    {
      "original_file": "video1.mp4",
      "audio_file": "audio/video1.wav",
      "video_file": "video/video1_video_only.mp4",
      "duration": 120.5
    }
  ],
  "total_pairs": 1
}
```

## Temporal Alignment

The system maintains temporal alignment through:

1. **Synchronized Extraction** - Audio and video are extracted from the same source simultaneously
2. **Metadata Timestamps** - Each file includes start_time and duration
3. **Frame/Sample Rates** - Original frame rates and sample rates are preserved
4. **Common Reference** - All timestamps reference the original video's timeline

### Verifying Alignment

The notebook includes an alignment verification section that displays:

- Duration of each media file
- Frame rates and sample rates
- Start times

This ensures audio and video remain synchronized for downstream processing.

## API Reference

### MediaProcessor Class

#### `__init__(input_dir, output_base_dir=None)`

Initialize the media processor.

#### `process_all(audio_format='wav', file_pattern='*.mp4')`

Process all matching files in the input directory.

#### `process_file(video_file, audio_format='wav')`

Process a single video file.

#### `get_aligned_pair(original_filename)`

Retrieve audio path, video path, and metadata for a processed file.

### Utility Functions

#### `check_dependencies()`

Check if ffmpeg and ffprobe are installed.

#### `get_processed_files_summary(metadata_dir)`

Get summary of all processed files.

#### `format_duration(seconds)`

Format duration as HH:MM:SS.

## Troubleshooting

### ffmpeg not found

- Ensure ffmpeg is installed and in your system PATH
- Test: `ffmpeg -version`
- Add ffmpeg to PATH or provide full path to executable

### No files found

- Check that MP4 files are in `input_data_buffer/`
- Verify file permissions
- Ensure files have `.mp4` extension

### Processing fails

- Check if input file is corrupted: `ffmpeg -i your_file.mp4`
- Ensure sufficient disk space
- Verify ffmpeg can read the codec

### Audio/Video out of sync

- This typically doesn't happen as we preserve original timing
- Check metadata for correct duration and frame rates
- Verify original file plays correctly

## Examples

### Process with different audio format

```python
processor.process_all(audio_format='mp3')
```

### Process only specific files

```python
processor.process_all(file_pattern='scene*.mp4')
```

### Custom output directory

```python
processor = MediaProcessor(
    input_dir='./videos',
    output_base_dir='./custom_output'
)
```

## Integration with Classification Pipeline

The processed audio and video files can be fed into:

- Audio classifier (`audio-classifier/`)
- Image classifier (`image-classifier/`)
- Post-inference fusion (`post_inference/`)

The metadata ensures temporal alignment during multi-modal classification.

## License

Part of the UrbanNoiseClassifier project.
