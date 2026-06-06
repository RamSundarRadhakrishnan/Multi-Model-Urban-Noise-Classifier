import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "inference"))
sys.path.append(str(ROOT / "image-classifier" / "video_inference"))

from cnn_bilstm_audio_inference import CNNBiLSTMBatchAudioInference
from video_frame_extractor import VideoFrameExtractor
from dual_yolo_inference import DualYOLOInference
from output_fusion import create_fusion_engine


CLASS_TO_VISUAL_MAP = {
    "crowd-noise": ["person", "people", "crowd"],
    "generator": ["truck", "engine", "machine", "machinery", "fan", "generator"],
    "motorvehicle-horn": ["car", "truck", "bus", "motorcycle", "vehicle", "taxi"],
    "mobile-music": ["person", "people", "guitar", "piano", "drum", "instrument"],
    "community-radio": ["person", "people", "radio"],
    "construction-site": ["excavator", "truck", "crane", "worker", "machinery", "machine", "construction", "bulldozer", "loader", "roller"],
    "motorvehicle-siren": ["car", "truck", "bus", "vehicle", "ambulance", "police", "fire"],
    "car-alarm": ["car", "vehicle", "truck"]
}


def normalize(value):
    return str(value).lower().replace("_", " ").replace("-", " ").strip()


def read_vggsound_csv(path):
    df = pd.read_csv(path, header=None, names=["youtube_id", "start_seconds", "vggsound_label", "split"])
    df["start_seconds"] = pd.to_numeric(df["start_seconds"], errors="coerce")
    df = df.dropna(subset=["start_seconds"])
    df["start_seconds"] = df["start_seconds"].astype(int)
    df["vggsound_label_norm"] = df["vggsound_label"].map(normalize)
    df["split"] = df["split"].map(lambda x: normalize(x))
    return df


def read_mapping(path, core_only):
    df = pd.read_csv(path)
    df["vggsound_label_norm"] = df["vggsound_label"].map(normalize)
    if core_only:
        df = df[df["use_for_core_eval"].map(normalize) == "yes"]
    return df


def build_video_index(videos_dir):
    suffixes = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    files = [p for p in Path(videos_dir).rglob("*") if p.suffix.lower() in suffixes]
    by_name = {p.name: p for p in files}
    by_stem = {p.stem: p for p in files}
    return files, by_name, by_stem


def locate_video(row, files, by_name, by_stem):
    youtube_id = str(row["youtube_id"])
    start = int(row["start_seconds"])
    stems = [
        f"{youtube_id}_{start}",
        f"{youtube_id}-{start}",
        f"{youtube_id}_{start:06d}",
        f"{youtube_id}-{start:06d}",
        youtube_id
    ]

    for stem in stems:
        if stem in by_stem:
            return by_stem[stem]

    for suffix in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        for stem in stems:
            name = stem + suffix
            if name in by_name:
                return by_name[name]

    start_text = str(start)
    for path in files:
        stem = path.stem
        if youtube_id in stem and start_text in stem:
            return path

    return None


def read_manifest_csv(manifest_csv, mapping_csv, split, samples_per_class):
    manifest_path = Path(manifest_csv).resolve()
    df = pd.read_csv(manifest_path)

    if "target_class" not in df.columns:
        mapping = pd.read_csv(mapping_csv)

        df["vggsound_label_norm"] = df["vggsound_label"].astype(str).str.lower().str.strip()
        mapping["vggsound_label_norm"] = mapping["vggsound_label"].astype(str).str.lower().str.strip()

        label_to_target = dict(zip(mapping["vggsound_label_norm"], mapping["target_class"]))
        df["target_class"] = df["vggsound_label_norm"].map(label_to_target)

        missing = df[df["target_class"].isna()]["vggsound_label"].drop_duplicates().tolist()
        if missing:
            raise RuntimeError("Missing mappings for labels: " + ", ".join(missing))

        df = df.drop(columns=["vggsound_label_norm"])

    required = {"clip_id", "vggsound_label", "target_class", "video_path"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")

    if "split" in df.columns and split:
        df = df[df["split"].map(normalize) == normalize(split)]

    base = manifest_path.parent

    def resolve_path(value):
        path = Path(value)

        if path.exists():
            return str(path.resolve())

        candidate = base / path
        if candidate.exists():
            return str(candidate.resolve())

        candidate = base / "videos_subset" / path.name
        if candidate.exists():
            return str(candidate.resolve())

        return None

    df["video_path"] = df["video_path"].map(resolve_path)
    df = df.dropna(subset=["video_path"])

    if samples_per_class is not None and samples_per_class > 0:
        df = df.groupby("target_class", group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
        )

    return df.reset_index(drop=True)


def extract_audio(manifest, audio_dir, sample_rate):
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_paths = []

    for _, row in manifest.iterrows():
        wav_path = audio_dir / f"{row['clip_id']}.wav"
        if not wav_path.exists():
            cmd = [
                "ffmpeg",
                "-y",
                "-i", row["video_path"],
                "-vn",
                "-ac", "1",
                "-ar", str(sample_rate),
                str(wav_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio_paths.append(str(wav_path))

    result = manifest.copy()
    result["audio_path"] = audio_paths
    return result


def visual_match(detected_class, relevant_classes):
    detected = normalize(detected_class)
    for relevant in relevant_classes:
        if normalize(relevant) in detected:
            return True
    return False


def detection_class(detection):
    if isinstance(detection, dict):
        return detection.get("class_name") or detection.get("label") or detection.get("name") or detection.get("class")
    return getattr(detection, "class_name", None)


def detection_confidence(detection):
    if isinstance(detection, dict):
        return float(detection.get("confidence", detection.get("score", detection.get("conf", 0.0))))
    return float(getattr(detection, "confidence", 0.0))


def collect_detections(video_result):
    detections = []
    if hasattr(video_result, "standard_detections"):
        detections.extend(video_result.standard_detections)
    if hasattr(video_result, "mocs_detections"):
        detections.extend(video_result.mocs_detections)
    if isinstance(video_result, dict):
        detections.extend(video_result.get("standard_detections", []))
        detections.extend(video_result.get("mocs_detections", []))
    return detections


def compute_visual_scores(video_result, class_names):
    scores = {name: 0.0 for name in class_names}

    for detection in collect_detections(video_result):
        cls = detection_class(detection)
        conf = detection_confidence(detection)
        if cls is None:
            continue

        for target_class in class_names:
            relevant = CLASS_TO_VISUAL_MAP.get(target_class, [])
            if visual_match(cls, relevant):
                scores[target_class] = max(scores[target_class], conf)

    is_construction = getattr(video_result, "is_construction_site", False)
    construction_conf = float(getattr(video_result, "construction_confidence", 0.0))

    if isinstance(video_result, dict):
        is_construction = video_result.get("is_construction_site", is_construction)
        construction_conf = float(video_result.get("construction_confidence", construction_conf))

    if "construction-site" in scores and is_construction:
        scores["construction-site"] = max(scores["construction-site"], construction_conf)

    return scores


def add_target_class_from_mapping(df, mapping_csv):
    if "target_class" in df.columns:
        return df

    mapping = pd.read_csv(mapping_csv)

    df = df.copy()
    mapping = mapping.copy()

    df["vggsound_label_norm_tmp"] = df["vggsound_label"].astype(str).str.lower().str.strip()
    mapping["vggsound_label_norm_tmp"] = mapping["vggsound_label"].astype(str).str.lower().str.strip()

    label_to_target = dict(zip(mapping["vggsound_label_norm_tmp"], mapping["target_class"]))
    df["target_class"] = df["vggsound_label_norm_tmp"].map(label_to_target)

    missing = df[df["target_class"].isna()]["vggsound_label"].drop_duplicates().tolist()
    if missing:
        raise RuntimeError("Missing target_class mappings for labels: " + ", ".join(missing))

    df = df.drop(columns=["vggsound_label_norm_tmp"])
    return df

def run_audio_inference(manifest, checkpoint, output_dir, sample_rate, batch_size, device, mapping_csv):
    audio_dir = output_dir / "audio"
    manifest = extract_audio(manifest, audio_dir, sample_rate)

    inferencer = CNNBiLSTMBatchAudioInference(
        checkpoint_path=checkpoint,
        sample_rate=sample_rate,
        batch_size=batch_size,
        device=device
    )

    df = inferencer.process_directory(
        input_dir=str(audio_dir),
        output_dir=str(output_dir / "audio_inference"),
        save_format="both"
    )

    df["clip_id"] = df["filename"].map(lambda x: Path(x).stem)

    merged = manifest.merge(
        df,
        on="clip_id",
        how="inner",
        suffixes=("", "_audio")
    )

    if "target_class" not in merged.columns:
        merged = add_target_class_from_mapping(merged, mapping_csv)

    if "target_class" not in merged.columns:
        candidates = [
            "target_class_x",
            "target_class_manifest",
            "target_class_audio"
        ]

        for candidate in candidates:
            if candidate in merged.columns:
                merged["target_class"] = merged[candidate]
                break

    if "target_class" not in merged.columns:
        raise RuntimeError(
            "target_class missing after audio merge. Columns are: "
            + ", ".join(merged.columns)
        )

    return merged, inferencer.label_names


def run_video_inference(manifest, yolo_standard_model, yolo_mocs_model, output_dir, frame_skip, conf_threshold, device):
    extractor = VideoFrameExtractor(frame_skip=frame_skip)
    yolo = DualYOLOInference(
        standard_model_path=yolo_standard_model,
        mocs_model_path=yolo_mocs_model,
        conf_threshold=conf_threshold,
        device=device
    )
    fusion_engine = create_fusion_engine(
        aggregation_method="weighted_average",
        min_detection_percentage=10.0,
        construction_threshold=0.3
    )

    video_results = {}

    for _, row in manifest.iterrows():
        frames = list(extractor.extract_frames(row["video_path"]))
        frame_results = yolo.infer_frames(frames)
        metadata = extractor.get_video_info(row["video_path"])
        result = fusion_engine.fuse_video_results(
            video_path=row["video_path"],
            frame_results=frame_results,
            video_metadata=metadata
        )
        video_results[row["clip_id"]] = result.to_dict()

    with open(output_dir / "video_inference_results.json", "w") as f:
        json.dump(video_results, f, indent=4)

    return video_results


def get_audio_scores(row, class_names):
    scores = {}

    for class_name in class_names:
        column = f"prob_{class_name}"
        if column in row and pd.notna(row[column]):
            scores[class_name] = float(row[column])
        else:
            scores[class_name] = 0.0

    return scores


def argmax_score(scores):
    return max(scores.items(), key=lambda item: item[1])[0]


def evaluate_predictions(df, prediction_column, labels):
    y_true = df["target_class"].tolist()
    y_pred = df[prediction_column].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1)
    }


def build_predictions(df, video_results, class_names, alpha):
    rows = []

    for _, row in df.iterrows():
        audio_scores = get_audio_scores(row, class_names)
        visual_scores = compute_visual_scores(video_results[row["clip_id"]], class_names)

        fusion_scores = {}
        for class_name in class_names:
            fusion_scores[class_name] = alpha * audio_scores[class_name] + (1.0 - alpha) * visual_scores[class_name]

        audio_pred = argmax_score(audio_scores)
        visual_pred = argmax_score(visual_scores)
        fusion_pred = argmax_score(fusion_scores)

        record = row.to_dict()
        record["audio_pred"] = audio_pred
        record["audio_conf"] = float(audio_scores[audio_pred])
        record["visual_pred"] = visual_pred
        record["visual_conf"] = float(visual_scores[visual_pred])
        record["fusion_pred"] = fusion_pred
        record["fusion_conf"] = float(fusion_scores[fusion_pred])
        record["alpha"] = alpha
        record["audio_correct"] = audio_pred == row["target_class"]
        record["visual_correct"] = visual_pred == row["target_class"]
        record["fusion_correct"] = fusion_pred == row["target_class"]
        record["audio_scores_json"] = json.dumps(audio_scores)
        record["visual_scores_json"] = json.dumps(visual_scores)
        record["fusion_scores_json"] = json.dumps(fusion_scores)

        rows.append(record)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--videos_dir", default=None)
    parser.add_argument("--manifest_csv", default=None)
    parser.add_argument("--mapping_csv", default="vggsound_urban_map.csv")
    parser.add_argument("--checkpoint", default="../cnn-bilstm-audio/best_cnn_bilstm_large.pt")
    parser.add_argument("--yolo_standard_model", default="yolov8n.pt")
    parser.add_argument("--yolo_mocs_model", default=None)
    parser.add_argument("--output_dir", default="vggsound_fusion_eval_output")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_batch_size", type=int, default=16)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--core_only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_csv:
        manifest = read_manifest_csv(
            manifest_csv=args.manifest_csv,
            mapping_csv=args.mapping_csv,
            split=args.split,
            samples_per_class=args.samples_per_class
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_dir / "vggsound_eval_manifest.csv", index=False)
    else:
        manifest = make_manifest(
            metadata_csv=args.metadata_csv,
            mapping_csv=args.mapping_csv,
            videos_dir=args.videos_dir,
            split=args.split,
            samples_per_class=args.samples_per_class,
            core_only=args.core_only,
            output_dir=output_dir
        )

    if len(manifest) == 0:
        raise RuntimeError("No matching VGGSound clips found. Check metadata CSV, mapping CSV, videos directory, and filename format.")

    audio_df, model_class_names = run_audio_inference(
        manifest=manifest,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        batch_size=args.audio_batch_size,
        device=args.device,
        mapping_csv=args.mapping_csv
    )

    eval_class_names = sorted(audio_df["target_class"].unique().tolist())

    video_results = run_video_inference(
        manifest=audio_df,
        yolo_standard_model=args.yolo_standard_model,
        yolo_mocs_model=args.yolo_mocs_model,
        output_dir=output_dir,
        frame_skip=args.frame_skip,
        conf_threshold=args.yolo_conf_threshold,
        device=args.device
    )

    predictions = build_predictions(
        df=audio_df,
        video_results=video_results,
        class_names=eval_class_names,
        alpha=args.alpha
    )

    predictions.to_csv(output_dir / "vggsound_fusion_predictions.csv", index=False)

    rows = []
    rows.append({"variant": "audio_only", **evaluate_predictions(predictions, "audio_pred", eval_class_names)})
    rows.append({"variant": "visual_only", **evaluate_predictions(predictions, "visual_pred", eval_class_names)})
    rows.append({"variant": f"late_fusion_alpha_{args.alpha}", **evaluate_predictions(predictions, "fusion_pred", eval_class_names)})

    ablation = pd.DataFrame(rows)
    ablation.to_csv(output_dir / "vggsound_ablation_table.csv", index=False)

    report = {
        "classes": eval_class_names,
        "num_samples": int(len(predictions)),
        "samples_per_class": predictions["target_class"].value_counts().to_dict(),
        "ablation": rows,
        "audio_classification_report": classification_report(
            predictions["target_class"],
            predictions["audio_pred"],
            labels=eval_class_names,
            zero_division=0,
            output_dict=True
        ),
        "visual_classification_report": classification_report(
            predictions["target_class"],
            predictions["visual_pred"],
            labels=eval_class_names,
            zero_division=0,
            output_dict=True
        ),
        "fusion_classification_report": classification_report(
            predictions["target_class"],
            predictions["fusion_pred"],
            labels=eval_class_names,
            zero_division=0,
            output_dict=True
        ),
        "fusion_confusion_matrix": confusion_matrix(
            predictions["target_class"],
            predictions["fusion_pred"],
            labels=eval_class_names
        ).tolist()
    }

    with open(output_dir / "vggsound_eval_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nAblation table")
    print(ablation.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()