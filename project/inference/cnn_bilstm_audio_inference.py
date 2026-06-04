import sys
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

CNN_BILSTM_DIR = Path(__file__).resolve().parents[1] / "cnn-bilstm-audio"
sys.path.append(str(CNN_BILSTM_DIR))

from model import CNNBiLSTMAudioClassifier


class CNNBiLSTMBatchAudioInference:
    def __init__(
        self,
        checkpoint_path,
        sample_rate=16000,
        n_mfcc=40,
        max_frames=None,
        batch_size=32,
        device="cuda"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.label_names = checkpoint["label_names"]
        self.class_to_id = checkpoint["class_to_id"]
        self.id_to_class = checkpoint["id_to_class"]
        self.num_classes = checkpoint.get("num_classes", len(self.label_names))
        self.max_frames = max_frames or checkpoint.get("max_frames", 174)

        self.model = CNNBiLSTMAudioClassifier(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _extract_mfcc(self, audio_path):
        waveform, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True
        )

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc
        )

        mfcc = mfcc.T

        if mfcc.shape[0] < self.max_frames:
            pad = self.max_frames - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode="constant")
        else:
            mfcc = mfcc[:self.max_frames, :]

        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        return mfcc.astype(np.float32)

    def _predict_batch(self, audio_paths):
        features = []

        for path in audio_paths:
            mfcc = self._extract_mfcc(path)
            features.append(mfcc)

        features = torch.tensor(np.stack(features), dtype=torch.float32)
        features = features.to(self.device)

        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)

        probs = probs.detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        return preds, confidences, probs

    def process_directory(self, input_dir, output_dir=None, save_format="both"):
        input_dir = Path(input_dir)

        audio_files = sorted(input_dir.glob("*.wav"))

        if len(audio_files) == 0:
            audio_files = sorted(input_dir.rglob("*.wav"))

        records = []

        for start in range(0, len(audio_files), self.batch_size):
            batch_files = audio_files[start:start + self.batch_size]
            preds, confidences, probs = self._predict_batch(batch_files)

            for audio_path, pred, confidence, prob_vector in zip(batch_files, preds, confidences, probs):
                label_name = self.label_names[int(pred)]

                record = {
                    "filename": audio_path.name,
                    "filepath": str(audio_path),
                    "predicted_class_id": int(pred),
                    "predicted_class_name": label_name,
                    "confidence": float(confidence),
                    "probability_vector": prob_vector.tolist()
                }

                for i, class_name in enumerate(self.label_names):
                    record[f"prob_{class_name}"] = float(prob_vector[i])

                records.append(record)

        df = pd.DataFrame(records)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_format in ["csv", "both"]:
                df.to_csv(output_dir / "cnn_bilstm_audio_inference.csv", index=False)

            if save_format in ["json", "both"]:
                with open(output_dir / "cnn_bilstm_audio_inference.json", "w") as f:
                    json.dump(records, f, indent=4)

        return df