"""
Audio Analysis Module
Calculates A-weighted noise levels and other acoustic metrics.
"""

import numpy as np
import wave
import struct
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import json


class AudioAnalyzer:
    """
    Analyze audio files to calculate A-weighted noise levels and other metrics.
    """
    
    # A-weighting filter coefficients (approximation for common sample rates)
    A_WEIGHTING_COEFFICIENTS = {
        44100: {
            # Simplified A-weighting for 44.1 kHz
            'frequencies': [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000],
            'weights': [-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8, 0.0, 0.6, 1.0, 1.2, 1.3, 1.2, 1.0, 0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3]
        }
    }
    
    def __init__(self):
        """Initialize the AudioAnalyzer."""
        pass
    
    def read_wav_file(self, filepath: Path) -> Tuple[np.ndarray, int, int]:
        """
        Read WAV file and return audio data.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            Tuple of (audio_data, sample_rate, num_channels)
        """
        with wave.open(str(filepath), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            # Read all frames
            frames = wav_file.readframes(num_frames)
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
                audio_data = np.frombuffer(frames, dtype=dtype)
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                dtype = np.int16
                audio_data = np.frombuffer(frames, dtype=dtype)
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                dtype = np.int32
                audio_data = np.frombuffer(frames, dtype=dtype)
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Reshape for stereo
            if num_channels > 1:
                audio_data = audio_data.reshape(-1, num_channels)
            
            return audio_data, sample_rate, num_channels
    
    def calculate_rms(self, audio_data: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) of audio signal.
        
        Args:
            audio_data: Audio samples as numpy array
            
        Returns:
            RMS value
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        rms = np.sqrt(np.mean(audio_data ** 2))
        return float(rms)
    
    def rms_to_db(self, rms: float, reference: float = 1.0) -> float:
        """
        Convert RMS to decibels.
        
        Args:
            rms: RMS value
            reference: Reference level (default 1.0 for full scale)
            
        Returns:
            dB value
        """
        if rms <= 0:
            return -np.inf
        
        db = 20 * np.log10(rms / reference)
        return float(db)
    
    def apply_a_weighting_approximation(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply simplified A-weighting to audio data using frequency-domain filtering.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            
        Returns:
            A-weighted audio data
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # FFT
        fft_data = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        
        # Create A-weighting filter
        a_weight = self._get_a_weighting_filter(freqs)
        
        # Apply filter in frequency domain
        weighted_fft = fft_data * a_weight
        
        # Inverse FFT
        weighted_audio = np.fft.irfft(weighted_fft, len(audio_data))
        
        return weighted_audio
    
    def _get_a_weighting_filter(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate A-weighting filter for given frequencies.
        
        Args:
            frequencies: Array of frequencies
            
        Returns:
            A-weighting filter values (linear scale)
        """
        # A-weighting formula
        f = frequencies
        f = np.maximum(f, 1e-10)  # Avoid division by zero
        
        # Standard A-weighting formula
        f2 = f ** 2
        c1 = 12194 ** 2
        c2 = 20.6 ** 2
        c3 = 107.7 ** 2
        c4 = 737.9 ** 2
        
        numerator = c1 * f2 * f2
        denominator = (f2 + c2) * np.sqrt((f2 + c3) * (f2 + c4)) * (f2 + c1)
        
        # A-weighting in dB
        a_db = 2.0 + 20 * np.log10(numerator / denominator)
        
        # Convert to linear scale
        a_linear = 10 ** (a_db / 20)
        
        return a_linear
    
    def calculate_a_weighted_spl(self, audio_data: np.ndarray, sample_rate: int, 
                                  reference_pressure: float = 2e-5) -> float:
        """
        Calculate A-weighted Sound Pressure Level (SPL) in dB(A).
        
        Note: This gives a relative measurement. For absolute SPL, calibration is needed.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            reference_pressure: Reference pressure (Pa), default is 20 μPa
            
        Returns:
            A-weighted SPL in dB(A)
        """
        # Apply A-weighting
        weighted_audio = self.apply_a_weighting_approximation(audio_data, sample_rate)
        
        # Calculate RMS
        rms = self.calculate_rms(weighted_audio)
        
        # Convert to dB (relative to full scale)
        # For actual SPL, this would need calibration
        db_fs = self.rms_to_db(rms, reference=1.0)
        
        # Adjust to approximate dB(A) scale
        # Note: This is a simplified calculation
        # Real-world SPL requires proper calibration
        db_a = db_fs + 94  # Approximate offset to dB scale
        
        return float(db_a)
    
    def calculate_leq(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Calculate equivalent continuous sound level (Leq) in dB(A).
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Leq in dB(A)
        """
        # Apply A-weighting
        weighted_audio = self.apply_a_weighting_approximation(audio_data, sample_rate)
        
        # Calculate mean square
        if len(weighted_audio.shape) > 1:
            weighted_audio = np.mean(weighted_audio, axis=1)
        
        mean_square = np.mean(weighted_audio ** 2)
        
        # Convert to dB
        if mean_square <= 0:
            return -np.inf
        
        leq = 10 * np.log10(mean_square) + 94  # Approximate offset
        
        return float(leq)
    
    def analyze_audio_clip(self, filepath: Path) -> Dict:
        """
        Perform comprehensive audio analysis on a clip.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Read audio file
            audio_data, sample_rate, num_channels = self.read_wav_file(filepath)
            
            # Calculate metrics
            rms = self.calculate_rms(audio_data)
            db_fs = self.rms_to_db(rms)
            a_weighted_spl = self.calculate_a_weighted_spl(audio_data, sample_rate)
            leq = self.calculate_leq(audio_data, sample_rate)
            
            # Calculate peak level
            peak = np.max(np.abs(audio_data))
            peak_db = self.rms_to_db(peak) if peak > 0 else -np.inf
            
            # Calculate crest factor
            crest_factor = peak / rms if rms > 0 else 0
            crest_factor_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0
            
            return {
                'rms': float(rms),
                'rms_db': float(db_fs),
                'peak': float(peak),
                'peak_db': float(peak_db),
                'a_weighted_spl': float(a_weighted_spl),
                'leq_a': float(leq),
                'crest_factor': float(crest_factor),
                'crest_factor_db': float(crest_factor_db),
                'sample_rate': int(sample_rate),
                'num_channels': int(num_channels),
                'duration': len(audio_data) / sample_rate
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'rms': 0.0,
                'a_weighted_spl': 0.0,
                'leq_a': 0.0
            }
    
    def analyze_audio_clips_batch(self, filepaths: List[Path]) -> List[Dict]:
        """
        Analyze multiple audio clips.
        
        Args:
            filepaths: List of paths to audio files
            
        Returns:
            List of analysis results
        """
        results = []
        for filepath in filepaths:
            result = self.analyze_audio_clip(filepath)
            result['filename'] = filepath.name
            results.append(result)
        
        return results
