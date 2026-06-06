import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "inference"))
sys.path.append(str(ROOT / "image-classifier" / "video_inference"))

from cnn_bilstm_audio_inference import CNNBiLSTMBatchAudioInference
from video_frame_extractor import VideoFrameExtractor
from dual_yolo_inference import DualYOLOInference
from output_fusion import create_fusion_engine
from vggsound_fusion_eval import read_manifest_csv, compute_visual_scores, get_audio_scores, argmax_score


def sync(device):
    if device.startswith("cuda"):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass


def extract_one_audio(video_path, wav_path, sample_rate):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        str(wav_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def summarize(values):
    s = pd.Series(values)
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max())
    }

def add_target_class_from_mapping(df, mapping_csv):
    if "target_class" in df.columns:
        return df

    mapping = pd.read_csv(mapping_csv)
    mapping.columns = mapping.columns.str.strip().str.replace("\ufeff", "", regex=False)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", default="../../datasets/vggsound/vggsound_subset_manifest.csv")
    parser.add_argument("--mapping_csv", default="vggsound_urban_map.csv")
    parser.add_argument("--checkpoint", default="../cnn-bilstm-audio/best_cnn_bilstm_large_v2.pt")
    parser.add_argument("--yolo_standard_model", default="yolov8n.pt")
    parser.add_argument("--yolo_mocs_model", default=None)
    parser.add_argument("--output_dir", default="vggsound_latency_benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--audio_batch_size", type=int, default=16)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_inference_dir = output_dir / "audio_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest_csv(
        manifest_csv=args.manifest_csv,
        mapping_csv=args.mapping_csv,
        split=args.split,
        samples_per_class=args.samples_per_class
    )

    manifest = add_target_class_from_mapping(manifest, args.mapping_csv)

    manifest.to_csv(output_dir / "latency_manifest.csv", index=False)

    audio_extract_times = []
    audio_paths = []

    for _, row in manifest.iterrows():
        wav_path = audio_dir / f"{row['clip_id']}.wav"
        t0 = time.perf_counter()
        extract_one_audio(row["video_path"], wav_path, args.sample_rate)
        t1 = time.perf_counter()
        audio_extract_times.append(t1 - t0)
        audio_paths.append(str(wav_path))

    manifest["audio_path"] = audio_paths

    sync(args.device)
    t0 = time.perf_counter()
    audio_model = CNNBiLSTMBatchAudioInference(
        checkpoint_path=args.checkpoint,
        sample_rate=args.sample_rate,
        batch_size=args.audio_batch_size,
        device=args.device
    )
    sync(args.device)
    audio_load_time = time.perf_counter() - t0

    sync(args.device)
    t0 = time.perf_counter()
    audio_df = audio_model.process_directory(
        input_dir=str(audio_dir),
        output_dir=str(audio_inference_dir),
        save_format="both"
    )
    sync(args.device)
    audio_inference_total = time.perf_counter() - t0
    audio_inference_per_clip = audio_inference_total / max(len(manifest), 1)

    audio_df["clip_id"] = audio_df["filename"].map(lambda x: Path(x).stem)

    merged = manifest.merge(
        audio_df,
        on="clip_id",
        how="inner",
        suffixes=("", "_audio")
    )

    if "target_class" not in merged.columns:
        if "target_class_audio" in merged.columns:
            merged["target_class"] = merged["target_class_audio"]
        elif "target_class_x" in merged.columns:
            merged["target_class"] = merged["target_class_x"]
        elif "target_class_y" in merged.columns:
            merged["target_class"] = merged["target_class_y"]
        elif "target_class" in manifest.columns:
            target_lookup = dict(zip(manifest["clip_id"], manifest["target_class"]))
            merged["target_class"] = merged["clip_id"].map(target_lookup)

    if "target_class" not in merged.columns:
        raise RuntimeError(
            "target_class missing after latency benchmark merge. Columns are: "
            + ", ".join(merged.columns)
        )

    sync(args.device)
    t0 = time.perf_counter()
    extractor = VideoFrameExtractor(frame_skip=args.frame_skip)
    yolo = DualYOLOInference(
        standard_model_path=args.yolo_standard_model,
        mocs_model_path=args.yolo_mocs_model,
        conf_threshold=args.yolo_conf_threshold,
        device=args.device
    )
    fusion_engine = create_fusion_engine(
        aggregation_method="weighted_average",
        min_detection_percentage=10.0,
        construction_threshold=0.3
    )
    sync(args.device)
    video_load_time = time.perf_counter() - t0

    rows = []
    video_results = {}

    for _, row in merged.iterrows():
        clip_id = row["clip_id"]

        t0 = time.perf_counter()
        frames = list(extractor.extract_frames(row["video_path"]))
        t1 = time.perf_counter()

        sync(args.device)
        t2 = time.perf_counter()
        frame_results = yolo.infer_frames(frames)
        sync(args.device)
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        metadata = extractor.get_video_info(row["video_path"])
        visual_result = fusion_engine.fuse_video_results(
            video_path=row["video_path"],
            frame_results=frame_results,
            video_metadata=metadata
        )
        visual_dict = visual_result.to_dict()
        t5 = time.perf_counter()

        video_results[clip_id] = visual_dict

        rows.append({
            "clip_id": clip_id,
            "target_class": row["target_class"],
            "video_path": row["video_path"],
            "num_frames_sampled": len(frames),
            "audio_extract_time_sec": audio_extract_times[len(rows)],
            "audio_inference_time_sec": audio_inference_per_clip,
            "frame_extraction_time_sec": t1 - t0,
            "yolo_inference_time_sec": t3 - t2,
            "visual_aggregation_time_sec": t5 - t4
        })

    class_names = sorted(merged["target_class"].unique().tolist())
    prediction_rows = []

    for _, row in merged.iterrows():
        t0 = time.perf_counter()

        audio_scores = get_audio_scores(row, class_names)
        visual_scores = compute_visual_scores(video_results[row["clip_id"]], class_names)

        fusion_scores = {}
        for class_name in class_names:
            fusion_scores[class_name] = args.alpha * audio_scores[class_name] + (1.0 - args.alpha) * visual_scores[class_name]

        audio_pred = argmax_score(audio_scores)
        visual_pred = argmax_score(visual_scores)
        fusion_pred = argmax_score(fusion_scores)

        t1 = time.perf_counter()

        prediction_rows.append({
            "clip_id": row["clip_id"],
            "audio_pred": audio_pred,
            "visual_pred": visual_pred,
            "fusion_pred": fusion_pred,
            "late_fusion_time_sec": t1 - t0
        })

    timing_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(prediction_rows)

    timing_df = timing_df.merge(pred_df, on="clip_id", how="inner")
    timing_df["end_to_end_time_sec"] = (
        timing_df["audio_extract_time_sec"]
        + timing_df["audio_inference_time_sec"]
        + timing_df["frame_extraction_time_sec"]
        + timing_df["yolo_inference_time_sec"]
        + timing_df["visual_aggregation_time_sec"]
        + timing_df["late_fusion_time_sec"]
    )

    timing_df.to_csv(output_dir / "latency_per_clip.csv", index=False)

    stages = [
        "audio_extract_time_sec",
        "audio_inference_time_sec",
        "frame_extraction_time_sec",
        "yolo_inference_time_sec",
        "visual_aggregation_time_sec",
        "late_fusion_time_sec",
        "end_to_end_time_sec"
    ]

    summary_rows = []

    for stage in stages:
        stats = summarize(timing_df[stage])
        summary_rows.append({
            "stage": stage,
            "mean_sec": stats["mean"],
            "median_sec": stats["median"],
            "std_sec": stats["std"],
            "min_sec": stats["min"],
            "max_sec": stats["max"],
            "throughput_clips_per_sec": 1.0 / stats["mean"] if stats["mean"] > 0 else None
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "latency_summary.csv", index=False)

    metadata = pd.DataFrame([
        {"item": "audio_model_load_time_sec", "value": audio_load_time},
        {"item": "video_model_load_time_sec", "value": video_load_time},
        {"item": "num_clips", "value": len(timing_df)},
        {"item": "samples_per_class", "value": args.samples_per_class},
        {"item": "frame_skip", "value": args.frame_skip},
        {"item": "device", "value": args.device},
        {"item": "alpha", "value": args.alpha}
    ])
    metadata.to_csv(output_dir / "latency_metadata.csv", index=False)

    print("\nLatency summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()