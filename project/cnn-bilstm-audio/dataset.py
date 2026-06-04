from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import torch

from datasets import Audio, load_dataset
from torch.utils.data import Dataset


class HFAudioMFCCDataset(Dataset):
    def __init__(
        self,
        dataset_name="Sunbird/urban-noise-uganda-61k",
        config_name="small",
        split="train",
        audio_column=None,
        label_column=None,
        sample_rate=16000,
        n_mfcc=40,
        n_mels=64,
        feature_type="mfcc",
        max_frames=174,
        target_classes=None
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.feature_type = feature_type

        self.ds = load_dataset(dataset_name, config_name, split=split)

        self.audio_column = audio_column or self._find_audio_column()
        self.label_column = label_column or self._find_label_column()

        self.ds = self.ds.cast_column(self.audio_column, Audio(decode=False))

        self.original_label_names = self._get_label_names()

        if target_classes is not None:
            target_set = set(target_classes)

            def keep_item(example):
                return self._label_to_name(example[self.label_column]) in target_set

            self.ds = self.ds.filter(keep_item)

            present = []
            for value in self.ds.unique(self.label_column):
                name = self._label_to_name(value)
                if name in target_set:
                    present.append(name)

            self.label_names = sorted(present)
        else:
            self.label_names = sorted([self._label_to_name(x) for x in self.ds.unique(self.label_column)])

        self.class_to_id = {name: idx for idx, name in enumerate(self.label_names)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}

    def _find_audio_column(self):
        for col in self.ds.column_names:
            if self.ds.features[col].__class__.__name__ == "Audio":
                return col
        for col in ["audio", "file", "filepath", "path", "wav"]:
            if col in self.ds.column_names:
                return col
        raise ValueError(f"Audio column not found. Columns: {self.ds.column_names}")

    def _find_label_column(self):
        for col in ["label", "labels", "class", "category", "class_name", "target"]:
            if col in self.ds.column_names:
                return col
        raise ValueError(f"Label column not found. Columns: {self.ds.column_names}")

    def _get_label_names(self):
        feature = self.ds.features[self.label_column]

        if hasattr(feature, "names") and feature.names is not None:
            return list(feature.names)

        return sorted([str(x) for x in self.ds.unique(self.label_column)])

    def _label_to_name(self, label):
        feature = self.ds.features[self.label_column]

        if hasattr(feature, "names") and feature.names is not None:
            return feature.names[int(label)]

        return str(label)

    def __len__(self):
        return len(self.ds)

    def _load_waveform(self, audio_obj):
        audio_bytes = audio_obj.get("bytes")
        path = audio_obj.get("path")

        if audio_bytes is not None:
            data, sr = sf.read(BytesIO(audio_bytes), dtype="float32")

            if data.ndim > 1:
                data = data.mean(axis=1)

            if sr != self.sample_rate:
                data = librosa.resample(
                    data,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )

            return data.astype(np.float32)

        if path is not None:
            waveform, sr = librosa.load(path, sr=self.sample_rate, mono=True)
            return waveform.astype(np.float32)

        raise ValueError("Audio object has neither bytes nor path")

    def _extract_features(self, waveform):
        if self.feature_type == "mfcc":
            features = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc
            )

        elif self.feature_type == "mfcc_delta":
            mfcc = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc
            )

            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(mfcc, order=2)

            features = np.concatenate([mfcc, delta, delta_delta], axis=0)

        elif self.feature_type == "logmel":
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels
            )

            features = librosa.power_to_db(mel, ref=np.max)

        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        features = features.T

        if features.shape[0] < self.max_frames:
            pad = self.max_frames - features.shape[0]
            features = np.pad(features, ((0, pad), (0, 0)), mode="constant")
        else:
            features = features[:self.max_frames, :]

        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        return features.astype(np.float32)

    def __getitem__(self, idx):
        item = self.ds[idx]

        audio_obj = item[self.audio_column]
        waveform = self._load_waveform(audio_obj)

        original_label = item[self.label_column]
        label_name = self._label_to_name(original_label)
        label_id = self.class_to_id[label_name]

        features = self._extract_features(waveform)
        return torch.tensor(features), torch.tensor(label_id, dtype=torch.long)