from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UrbanNoiseMFCCDataset(Dataset):
    def __init__(
        self,
        csv_path,
        audio_dir=None,
        audio_column=None,
        label_column=None,
        sample_rate=16000,
        n_mfcc=40,
        max_frames=174,
        target_classes=None
    ):
        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_frames = max_frames

        self.df = pd.read_csv(self.csv_path)

        self.audio_column = audio_column or self._find_audio_column()
        self.label_column = label_column or self._find_label_column()

        if target_classes is not None:
            self.df = self.df[self.df[self.label_column].isin(target_classes)].reset_index(drop=True)

        self.classes = sorted(self.df[self.label_column].unique().tolist())
        self.class_to_id = {name: idx for idx, name in enumerate(self.classes)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}

    def _find_audio_column(self):
        candidates = ["filepath", "file_path", "path", "filename", "file", "audio", "wav"]
        for col in candidates:
            if col in self.df.columns:
                return col
        raise ValueError(f"Could not find audio column. Columns found: {self.df.columns.tolist()}")

    def _find_label_column(self):
        candidates = ["label", "class", "category", "target", "class_name"]
        for col in candidates:
            if col in self.df.columns:
                return col
        raise ValueError(f"Could not find label column. Columns found: {self.df.columns.tolist()}")

    def __len__(self):
        return len(self.df)

    def _get_audio_path(self, value):
        path = Path(str(value))

        if path.is_absolute():
            return path

        if self.audio_dir is not None:
            return self.audio_dir / path

        return self.csv_path.parent / path

    def _extract_mfcc(self, audio_path):
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc
        )

        mfcc = mfcc.T

        if mfcc.shape[0] < self.max_frames:
            pad_width = self.max_frames - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfcc = mfcc[:self.max_frames, :]

        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        return mfcc.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio_path = self._get_audio_path(row[self.audio_column])
        label_name = row[self.label_column]
        label_id = self.class_to_id[label_name]

        mfcc = self._extract_mfcc(audio_path)

        return torch.tensor(mfcc), torch.tensor(label_id, dtype=torch.long)