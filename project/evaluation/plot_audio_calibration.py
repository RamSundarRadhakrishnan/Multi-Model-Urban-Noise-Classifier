"""Create calibration figures from the final CNN--BiLSTM held-out test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[2]
PREDICTIONS_CSV = (
    REPO_DIR
    / "project"
    / "cnn-bilstm-audio"
    / "metrics_cnn_bilstm_large_v2"
    / "predictions.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "paper_figures"
N_BINS = 10


def calibration_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Return confidence-bin statistics and the expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_confidence": np.nan,
                    "accuracy": np.nan,
                    "ece_contribution": 0.0,
                }
            )
            continue

        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        contribution = (count / len(confidence)) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "ece_contribution": contribution,
            }
        )

    return pd.DataFrame(rows)


def plot_reliability(bin_stats: pd.DataFrame, output_path: Path) -> None:
    populated = bin_stats.dropna(subset=["mean_confidence", "accuracy"])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect calibration")
    ax.plot(
        populated["mean_confidence"],
        populated["accuracy"],
        marker="o",
        color="#4477AA",
        linewidth=2,
        markersize=6,
        label="CNN--BiLSTM",
    )

    for _, row in populated.iterrows():
        ax.annotate(
            str(int(row["count"])),
            (row["mean_confidence"], row["accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(confidence: np.ndarray, correct: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(
        confidence[correct],
        bins=bins,
        alpha=0.75,
        color="#4477AA",
        label="Correct prediction",
    )
    ax.hist(
        confidence[~correct],
        bins=bins,
        alpha=0.75,
        color="#EE7733",
        label="Incorrect prediction",
    )

    ax.set_xlabel("Top-class predicted confidence")
    ax.set_ylabel("Number of test samples")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_CSV)

    probability_columns = [column for column in predictions.columns if column.startswith("prob_")]
    if not probability_columns:
        raise ValueError("No probability columns beginning with 'prob_' were found.")

    confidence = predictions[probability_columns].max(axis=1).to_numpy(dtype=float)
    correct = (
        predictions["actual_label"].to_numpy() == predictions["predicted_label"].to_numpy()
    )

    bin_stats = calibration_bins(confidence, correct, N_BINS)
    ece = float(bin_stats["ece_contribution"].sum())
    summary = {
        "dataset": "Held-out Urban Noise Uganda 61K test split",
        "num_samples": int(len(predictions)),
        "num_bins": N_BINS,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "expected_calibration_error": ece,
    }

    bin_stats.to_csv(OUTPUT_DIR / "audio_calibration_bin_statistics.csv", index=False)
    with open(OUTPUT_DIR / "audio_calibration_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plot_reliability(bin_stats, OUTPUT_DIR / "audio_confidence_reliability.png")
    plot_confidence_histogram(
        confidence,
        correct,
        OUTPUT_DIR / "audio_confidence_correct_incorrect_histogram.png",
    )

    print(f"ECE ({N_BINS} bins): {ece:.4f}")
    print(f"Saved figures and summaries to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
