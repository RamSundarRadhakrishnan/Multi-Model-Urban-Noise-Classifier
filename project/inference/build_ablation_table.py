import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_audio_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def summarize_audio_model(metrics):
    return {
        "experiment": "CNN-BiLSTM audio-only",
        "evaluation_type": "supervised audio test",
        "num_samples": metrics.get("num_test_samples"),
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_precision": metrics.get("weighted_precision"),
        "weighted_recall": metrics.get("weighted_recall"),
        "weighted_f1": metrics.get("weighted_f1"),
        "mean_confidence": None,
        "high_confidence_rate": None,
        "matched_visual_rate": None,
        "audio_only_fallback_rate": None,
        "violation_count": None,
        "notes": "Held-out UrbanNoiseUganda61K test split"
    }


def summarize_operational_ablation(df, variant):
    if variant == "audio_only":
        score_col = "fusion_audio_confidence"
        experiment = "CNN-BiLSTM audio-only operational"
        notes = "Uses only audio confidence from integrated pipeline events"
    elif variant == "video_only":
        score_col = "fusion_visual_confidence"
        experiment = "YOLO video-only operational"
        notes = "Uses only matched visual confidence where available"
    else:
        score_col = "fusion_score"
        experiment = "CNN-BiLSTM + YOLO late fusion"
        notes = "Decision-level late fusion with audio-only fallback"

    if score_col not in df.columns:
        raise ValueError(f"Missing required column: {score_col}")

    scores = df[score_col].fillna(0.0)

    fusion_mode_col = find_column(df, ["fusion_mode"])
    violation_col = find_column(df, ["is_compliant", "compliant"])
    noise_exceeded_col = find_column(df, ["noise_exceeded"])
    visual_col = find_column(df, ["fusion_visual_confidence"])

    if fusion_mode_col:
        matched_visual_rate = (df[fusion_mode_col] == "audio_visual_late_fusion").mean()
        fallback_rate = (df[fusion_mode_col] == "audio_only_fallback").mean()
    elif visual_col:
        matched_visual_rate = (df[visual_col].fillna(0.0) > 0.0).mean()
        fallback_rate = 1.0 - matched_visual_rate
    else:
        matched_visual_rate = None
        fallback_rate = None

    if violation_col == "is_compliant":
        violation_count = int((df[violation_col] == False).sum())
    elif violation_col == "compliant":
        violation_count = int((df[violation_col] == False).sum())
    elif noise_exceeded_col:
        violation_count = int((df[noise_exceeded_col] == True).sum())
    else:
        violation_count = None

    return {
        "experiment": experiment,
        "evaluation_type": "operational pipeline ablation",
        "num_samples": len(df),
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
        "mean_confidence": float(scores.mean()),
        "high_confidence_rate": float((scores >= 0.8).mean()),
        "matched_visual_rate": None if matched_visual_rate is None else float(matched_visual_rate),
        "audio_only_fallback_rate": None if fallback_rate is None else float(fallback_rate),
        "violation_count": violation_count,
        "notes": notes
    }


def save_markdown_table(df, path):
    with open(path, "w") as f:
        f.write(df.to_markdown(index=False))


def plot_confidence_comparison(df, output_path):
    plot_df = df[df["mean_confidence"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["experiment"], plot_df["mean_confidence"])

    ax.set_ylabel("Mean confidence")
    ax.set_title("Operational confidence comparison")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_high_confidence_rate(df, output_path):
    plot_df = df[df["high_confidence_rate"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(plot_df["experiment"], plot_df["high_confidence_rate"])

    ax.set_ylabel("Rate")
    ax.set_title("High-confidence detection rate")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fusion_modes(events_df, output_path):
    if "fusion_mode" not in events_df.columns:
        return

    counts = events_df["fusion_mode"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts.index, counts.values)

    ax.set_ylabel("Event count")
    ax.set_title("Fusion mode distribution")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_metrics", required=True)
    parser.add_argument("--events_csv", required=True)
    parser.add_argument("--output_dir", default="ablation_results")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_metrics = load_audio_metrics(args.audio_metrics)
    events_df = pd.read_csv(args.events_csv)

    rows = [
        summarize_audio_model(audio_metrics),
        summarize_operational_ablation(events_df, "audio_only"),
        summarize_operational_ablation(events_df, "video_only"),
        summarize_operational_ablation(events_df, "fusion")
    ]

    ablation_df = pd.DataFrame(rows)

    ablation_df.to_csv(output_dir / "ablation_table.csv", index=False)
    save_markdown_table(ablation_df, output_dir / "ablation_table.md")

    plot_confidence_comparison(ablation_df, output_dir / "ablation_mean_confidence.png")
    plot_high_confidence_rate(ablation_df, output_dir / "ablation_high_confidence_rate.png")
    plot_fusion_modes(events_df, output_dir / "fusion_mode_distribution.png")

    print(ablation_df)
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()