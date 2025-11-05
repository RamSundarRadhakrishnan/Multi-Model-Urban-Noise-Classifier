# Batch Inference for Fine-Tuned SpeechBrain Model

This module provides batch inference capabilities for the fine-tuned SpeechBrain audio classifier.

## Features

- 🎵 **Batch Processing**: Process entire directories of audio files efficiently
- 📊 **Probability Vectors**: Store complete probability distributions for each prediction
- 💾 **Multiple Output Formats**: Save results as CSV, NumPy arrays, and JSON
- 🚀 **GPU Support**: Automatic GPU detection and utilization
- 📈 **Progress Tracking**: Visual progress bars during processing

## Files

- `inference_utils.py` - Core utility functions for preprocessing and model inference
- `batch_inference.py` - Batch processing logic and result management
- `batch_inference_demo.ipynb` - Interactive notebook demonstrating usage

## Quick Start

### 1. Basic Usage

```python
from batch_inference import BatchAudioInference

# Initialize processor
processor = BatchAudioInference(
    pretrained_model_path="pretrained_urbansound8k",
    fine_tuned_weights_path="fine_tuned_urbansound8k_8class.pth",
    num_classes=8,
    class_names=["class_0", "class_1", ...],
    batch_size=8
)

# Process directory
results = processor.process_directory(
    input_dir="path/to/audio/files",
    output_dir="results",
    save_format='both'  # CSV and NumPy
)
```

### 2. Process Single File

```python
# Process single audio file
result = processor.process_single_file("path/to/audio.wav")

print(f"Predicted: {result['predicted_class_name']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probability_vector']}")
```

### 3. Load Saved Results

```python
from batch_inference import load_results

# Load results from previous run
df, prob_matrix, metadata = load_results("results")

print(f"Loaded {len(df)} predictions")
print(f"Probability matrix shape: {prob_matrix.shape}")
```

## Installation

Ensure you have the required dependencies:

```bash
pip install torch torchaudio speechbrain numpy pandas tqdm
```

## Configuration Parameters

### BatchAudioInference

- `pretrained_model_path` (str): Path to SpeechBrain pretrained model directory
- `fine_tuned_weights_path` (str): Path to fine-tuned model weights (.pth file)
- `num_classes` (int): Number of output classes
- `class_names` (List[str], optional): List of class names
- `sample_rate` (int): Target sample rate (default: 16000 Hz)
- `max_length` (int): Maximum audio length in samples (default: 160000 = 10 seconds)
- `batch_size` (int): Batch size for processing (default: 8)
- `device` (str, optional): Device for inference ('cuda' or 'cpu', auto-detected if None)

## Output Formats

### CSV Format

- Contains predictions, confidence scores, and individual class probabilities
- Columns: `filename`, `filepath`, `predicted_class_id`, `predicted_class_name`, `confidence`, `prob_class_0`, `prob_class_1`, ...

### NumPy Format

- `probability_vectors_<timestamp>.npy`: 2D array of shape (n_files, n_classes)
- Efficient storage for probability distributions

### JSON Format

- `metadata_<timestamp>.json`: Contains filenames, predictions, and metadata
- Structured format for easy loading and analysis

## Example Workflow

```python
import pandas as pd
import numpy as np
from batch_inference import BatchAudioInference, load_results

# 1. Initialize processor
processor = BatchAudioInference(
    pretrained_model_path="pretrained_urbansound8k",
    fine_tuned_weights_path="fine_tuned_urbansound8k_8class.pth",
    num_classes=8,
    class_names=[
        "air_conditioner", "car_horn", "children_playing",
        "dog_bark", "drilling", "engine_idling",
        "gun_shot", "siren"
    ]
)

# 2. Process audio files
results = processor.process_directory(
    input_dir="test_audio",
    output_dir="inference_results",
    save_format='both'
)

# 3. Analyze results
print(f"Processed {len(results)} files")
print(results['predicted_class_name'].value_counts())

# 4. Load and use probability vectors
df, prob_matrix, metadata = load_results("inference_results")

# Find predictions with high confidence
high_conf = df[df['confidence'] >= 0.9]
print(f"{len(high_conf)} high-confidence predictions")

# Analyze probability distributions
avg_probs = prob_matrix.mean(axis=0)
for i, class_name in enumerate(metadata['class_names']):
    print(f"{class_name}: {avg_probs[i]:.4f}")
```

## Performance Tips

1. **Batch Size**: Increase for faster processing on GPUs (e.g., 16-32)
2. **GPU Memory**: Reduce batch size if running out of memory
3. **Audio Length**: Shorter `max_length` = faster processing
4. **File Format**: WAV files are fastest, avoid conversions

## Error Handling

The batch processor handles common errors gracefully:

- **Invalid audio files**: Skipped with warning message
- **Corrupted files**: Logged and continued
- **Missing files**: Reported but processing continues

## Advanced Usage

### Custom Preprocessing

```python
from inference_utils import AudioPreprocessor

# Create custom preprocessor
preprocessor = AudioPreprocessor(
    sample_rate=22050,  # Different sample rate
    max_length=220500   # 10 seconds at 22.05kHz
)
```

### Filtering Results

```python
# Filter by confidence threshold
high_confidence = results[results['confidence'] >= 0.8]

# Filter by predicted class
specific_class = results[results['predicted_class_name'] == 'dog_bark']

# Export filtered results
high_confidence.to_csv('high_confidence_only.csv', index=False)
```

## Troubleshooting

### Issue: Out of memory

**Solution**: Reduce `batch_size` parameter

### Issue: Slow processing

**Solution**: Ensure GPU is available and increase `batch_size`

### Issue: Import errors

**Solution**: Install required packages: `pip install speechbrain torch torchaudio`

### Issue: Model not found

**Solution**: Verify paths to pretrained model and fine-tuned weights

## License

This module is part of the UrbanNoiseClassifier project.
