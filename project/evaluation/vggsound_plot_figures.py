import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


LABELS = [
    "construction-site",
    "crowd-noise",
    "motorvehicle-horn",
    "motorvehicle-siren",
]

DISPLAY_LABELS = [
    "Construction\nsite",
    "Crowd\nnoise",
    "Motor-vehicle\nhorn",
    "Motor-vehicle\nsiren",
]


def plot_ablation(ablation_path, output_path):
    df = pd.read_csv(ablation_path)
    methods = ["Audio only", "Visual only", "Late fusion\n($\\alpha=0.5$)"]
    accuracy = df["accuracy"].to_numpy() * 100
    macro_f1 = df["macro_f1"].to_numpy() * 100
    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars_accuracy = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#4477AA")
    bars_f1 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#EE7733")

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, methods)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    for bars in (bars_accuracy, bars_f1):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(predictions_path, output_path):
    df = pd.read_csv(predictions_path)
    matrix = confusion_matrix(
        df["target_class"],
        df["fusion_pred"],
        labels=LABELS,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion of ground-truth class")

    ax.set_xticks(np.arange(len(LABELS)), DISPLAY_LABELS)
    ax.set_yticks(np.arange(len(LABELS)), DISPLAY_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="vggsound_fusion_alpha_0.5")
    parser.add_argument("--output_dir", default="paper_figures")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation(
        input_dir / "vggsound_ablation_table.csv",
        output_dir / "vggsound_fusion_ablation.png",
    )
    plot_confusion(
        input_dir / "vggsound_fusion_predictions.csv",
        output_dir / "vggsound_fusion_confusion_matrix_normalized.png",
    )


if __name__ == "__main__":
    main()