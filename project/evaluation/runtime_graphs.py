from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "vggsound_latency_benchmark" / "latency_summary.csv"
OUTPUT_DIR = BASE_DIR / "paper_figures"
OUTPUT_FILE = OUTPUT_DIR / "runtime_breakdown.png"

STAGE_LABELS = {
    "audio_extract_time_sec": "Audio feature\nextraction",
    "audio_inference_time_sec": "CNN--BiLSTM\naudio inference",
    "frame_extraction_time_sec": "Video frame\nextraction",
    "yolo_inference_time_sec": "YOLOv8n\ninference",
    "visual_aggregation_time_sec": "Visual score\naggregation",
    "late_fusion_time_sec": "Late fusion",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    latency = pd.read_csv(INPUT_CSV)
    latency = latency[latency["stage"].isin(STAGE_LABELS)].copy()
    latency["label"] = latency["stage"].map(STAGE_LABELS)
    latency["mean_ms"] = latency["mean_sec"] * 1000

    # Keep the pipeline order used by the benchmark.
    stage_order = list(STAGE_LABELS.keys())
    latency["stage"] = pd.Categorical(
        latency["stage"], categories=stage_order, ordered=True
    )
    latency = latency.sort_values("stage")

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(
        latency["label"],
        latency["mean_ms"],
        color="#2E8B57",
        edgecolor="black",
        linewidth=0.7,
    )

    for bar, value in zip(bars, latency["mean_ms"]):
        label = f"{value:.2f} ms" if value >= 0.1 else f"{value:.3f} ms"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Mean processing time per clip (ms)")
    ax.set_title("Runtime breakdown of the late-fusion pipeline")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()