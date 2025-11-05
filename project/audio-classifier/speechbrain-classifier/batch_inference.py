"""
Batch processor for audio inference with SpeechBrain model.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from datetime import datetime

from inference_utils import AudioPreprocessor, ModelInference, get_audio_files


class BatchAudioInference:
    """Handles batch inference on audio files."""
    
    def __init__(
        self,
        pretrained_model_path: str,
        fine_tuned_weights_path: str,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        sample_rate: int = 16000,
        max_length: int = 160000,
        batch_size: int = 8,
        device: str = None
    ):
        """
        Initialize batch inference processor.
        
        Args:
            pretrained_model_path: Path to pretrained SpeechBrain model directory
            fine_tuned_weights_path: Path to fine-tuned model weights (.pth file)
            num_classes: Number of output classes
            class_names: List of class names (optional)
            sample_rate: Target sample rate (default: 16000 Hz)
            max_length: Maximum audio length in samples
            batch_size: Batch size for processing
            device: Device to run inference on
        """
        self.preprocessor = AudioPreprocessor(sample_rate, max_length)
        self.model = ModelInference(
            pretrained_model_path,
            fine_tuned_weights_path,
            num_classes,
            device
        )
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [f"class_{i}" for i in range(num_classes)]
        self.batch_size = batch_size
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_format: str = 'both'  # 'csv', 'npy', or 'both'
    ) -> pd.DataFrame:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save results
            save_format: Output format ('csv', 'npy', or 'both')
            
        Returns:
            DataFrame with results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all audio files
        audio_files = get_audio_files(input_dir)
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        if len(audio_files) == 0:
            print("No audio files found!")
            return pd.DataFrame()
        
        # Store results
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(audio_files), self.batch_size), desc="Processing batches"):
            batch_files = audio_files[i:i + self.batch_size]
            batch_waveforms = []
            
            # Load and preprocess batch
            for filepath in batch_files:
                try:
                    waveform = self.preprocessor.load_and_preprocess(filepath)
                    batch_waveforms.append(waveform)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    # Add dummy waveform to maintain batch consistency
                    batch_waveforms.append(torch.zeros(self.preprocessor.max_length))
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_waveforms)
            
            # Perform inference
            prob_vectors, predicted_classes = self.model.predict_batch(batch_tensor)
            
            # Store results
            for j, filepath in enumerate(batch_files):
                filename = os.path.basename(filepath)
                result = {
                    'filename': filename,
                    'filepath': filepath,
                    'predicted_class_id': int(predicted_classes[j]),
                    'predicted_class_name': self.class_names[predicted_classes[j]],
                    'confidence': float(prob_vectors[j][predicted_classes[j]]),
                    'probability_vector': prob_vectors[j]
                }
                
                # Add individual class probabilities
                for k, class_name in enumerate(self.class_names):
                    result[f'prob_{class_name}'] = float(prob_vectors[j][k])
                
                results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_format in ['csv', 'both']:
            csv_path = output_path / f'inference_results_{timestamp}.csv'
            # Save CSV without probability_vector column (too large)
            df_csv = df.drop(columns=['probability_vector'])
            df_csv.to_csv(csv_path, index=False)
            print(f"✅ Saved CSV results to: {csv_path}")
        
        if save_format in ['npy', 'both']:
            # Save probability vectors as numpy array
            npy_path = output_path / f'probability_vectors_{timestamp}.npy'
            prob_matrix = np.stack(df['probability_vector'].values)
            np.save(npy_path, prob_matrix)
            print(f"✅ Saved probability vectors to: {npy_path}")
            
            # Save metadata (filenames and predictions)
            metadata_path = output_path / f'metadata_{timestamp}.json'
            metadata = {
                'filenames': df['filename'].tolist(),
                'filepaths': df['filepath'].tolist(),
                'predicted_class_ids': df['predicted_class_id'].tolist(),
                'predicted_class_names': df['predicted_class_name'].tolist(),
                'confidences': df['confidence'].tolist(),
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'timestamp': timestamp
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Saved metadata to: {metadata_path}")
        
        return df
    
    def process_single_file(self, filepath: str) -> Dict:
        """
        Process a single audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary with results
        """
        # Preprocess
        waveform = self.preprocessor.load_and_preprocess(filepath)
        
        # Perform inference
        prob_vector, predicted_class = self.model.predict(waveform)
        
        # Prepare result
        result = {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'predicted_class_id': int(predicted_class),
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': float(prob_vector[predicted_class]),
            'probability_vector': prob_vector
        }
        
        # Add individual class probabilities
        for k, class_name in enumerate(self.class_names):
            result[f'prob_{class_name}'] = float(prob_vector[k])
        
        return result


def load_results(output_dir: str, timestamp: str = None) -> tuple:
    """
    Load saved inference results.
    
    Args:
        output_dir: Directory containing saved results
        timestamp: Specific timestamp to load (optional, loads latest if None)
        
    Returns:
        Tuple of (DataFrame, probability_matrix, metadata)
    """
    output_path = Path(output_dir)
    
    if timestamp:
        csv_file = output_path / f'inference_results_{timestamp}.csv'
        npy_file = output_path / f'probability_vectors_{timestamp}.npy'
        meta_file = output_path / f'metadata_{timestamp}.json'
    else:
        # Find latest files
        csv_files = list(output_path.glob('inference_results_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No results found in {output_dir}")
        
        csv_file = sorted(csv_files)[-1]
        timestamp = csv_file.stem.split('_', 2)[-1]
        npy_file = output_path / f'probability_vectors_{timestamp}.npy'
        meta_file = output_path / f'metadata_{timestamp}.json'
    
    # Load data
    df = pd.read_csv(csv_file)
    prob_matrix = np.load(npy_file) if npy_file.exists() else None
    
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    return df, prob_matrix, metadata
