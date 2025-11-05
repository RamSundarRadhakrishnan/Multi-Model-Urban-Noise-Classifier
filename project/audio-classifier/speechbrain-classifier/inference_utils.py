"""
Utility functions for SpeechBrain model inference.
"""
import os
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class AudioPreprocessor:
    """Handles audio preprocessing for inference."""
    
    def __init__(self, sample_rate: int = 16000, max_length: int = 160000):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate (default: 16000 Hz)
            max_length: Maximum audio length in samples (default: 160000 = 10 seconds at 16kHz)
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
    
    def load_and_preprocess(self, filepath: str) -> torch.Tensor:
        """
        Load and preprocess an audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Preprocessed audio waveform tensor
        """
        # Load audio
        waveform, sr = torchaudio.load(filepath)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to target sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate to fixed length
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Return as 1D tensor
        return waveform.squeeze(0)


class ModelInference:
    """Handles model loading and inference."""
    
    def __init__(
        self,
        pretrained_model_path: str,
        fine_tuned_weights_path: str,
        num_classes: int,
        device: str = None
    ):
        """
        Initialize inference model.
        
        Args:
            pretrained_model_path: Path to pretrained SpeechBrain model directory
            fine_tuned_weights_path: Path to fine-tuned model weights (.pth file)
            num_classes: Number of output classes
            device: Device to run inference on ('cuda' or 'cpu', default: auto-detect)
        """
        from speechbrain.inference import EncoderClassifier
        from torch import nn
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load base pretrained model
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/urbansound8k_ecapa",
            savedir=pretrained_model_path
        )
        
        # Replace classifier head
        n_features = 192  # ECAPA-TDNN output dimension
        new_classifier = nn.Sequential(
            nn.Linear(n_features, num_classes)
        )
        self.model.mods.classifier = new_classifier
        
        # Load fine-tuned weights
        self.model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, waveform: torch.Tensor) -> Tuple[np.ndarray, int]:
        """
        Perform inference on a single audio waveform.
        
        Args:
            waveform: Preprocessed audio waveform tensor
            
        Returns:
            Tuple of (probability vector, predicted class)
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            waveform = waveform.to(self.device)
            
            # Extract features
            feats = self.model.mods.compute_features(waveform)
            
            # Get embeddings
            embeddings = self.model.mods.embedding_model(feats)
            
            # Get logits
            logits = self.model.mods.classifier(embeddings.squeeze(1))
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # Convert to numpy
            prob_vector = probabilities.cpu().numpy()[0]
            
        return prob_vector, predicted_class
    
    def predict_batch(self, waveforms: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a batch of audio waveforms.
        
        Args:
            waveforms: Batch of preprocessed audio waveform tensors [batch_size, time]
            
        Returns:
            Tuple of (probability vectors [batch_size, num_classes], predicted classes [batch_size])
        """
        with torch.no_grad():
            waveforms = waveforms.to(self.device)
            
            # Extract features
            feats = self.model.mods.compute_features(waveforms)
            
            # Get embeddings
            embeddings = self.model.mods.embedding_model(feats)
            
            # Get logits
            logits = self.model.mods.classifier(embeddings.squeeze(1))
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Get predicted classes
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Convert to numpy
            prob_vectors = probabilities.cpu().numpy()
            predicted_classes = predicted_classes.cpu().numpy()
            
        return prob_vectors, predicted_classes


def get_audio_files(directory: str, extensions: List[str] = ['.wav', '.mp3', '.flac', '.ogg']) -> List[str]:
    """
    Get all audio files from a directory.
    
    Args:
        directory: Path to directory containing audio files
        extensions: List of valid audio file extensions
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    directory = Path(directory)
    
    for ext in extensions:
        # Case-insensitive search using glob pattern
        audio_files.extend(directory.glob(f'*{ext}'))
        audio_files.extend(directory.glob(f'*{ext.upper()}'))
    
    # Remove duplicates (in case files are found via both lowercase and uppercase patterns)
    audio_files_unique = list(set(audio_files))
    
    return sorted([str(f) for f in audio_files_unique])
