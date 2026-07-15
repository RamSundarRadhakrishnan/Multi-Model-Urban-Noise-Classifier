"""Generate paper figures for the frozen-branch learned-fusion evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


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

METHOD_LABELS = {
    "audio_only": "Audio\nonly",
    "visual_only": "Visual\nonly",
    "fixed_equal_weight_fusion": "Fixed late\nfusion",
    "learned_logistic_regression_fusion": "Learned late\nfusion",
}


def add_value_labels(bars, ax, fontsize=8):
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=fontsize,
        )


def plot_ablation(ablation_path, output_path):
    df = pd.read_csv(ablation_path)
    df = df[df["variant"].isin(METHOD_LABELS)].copy()
    df["label"] = df["variant"].map(METHOD_LABELS)
    df["order"] = df["variant"].map({key: index for index, key in enumerate(METHOD_LABELS)})
    df = df.sort_values("order")

    x = np.arange(len(df))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    accuracy_bars = ax.bar(
        x - width / 2, df["accuracy"] * 100, width, label="Accuracy", color="#4477AA"
    )
    f1_bars = ax.bar(
        x + width / 2, df["macro_f1"] * 100, width, label="Macro-F1", color="#2E8B57"
    )

    add_value_labels(accuracy_bars, ax)
    add_value_labels(f1_bars, ax)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, df["label"])
    ax.set_ylim(0, 65)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(predictions_path, output_path):
    df = pd.read_csv(predictions_path)
    matrix = confusion_matrix(
        df["target_class"],
        df["learned_fusion_pred"],
        labels=LABELS,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colourbar.set_label("Proportion of ground-truth class")
    ax.set_xticks(np.arange(len(LABELS)), DISPLAY_LABELS)
    ax.set_yticks(np.arange(len(LABELS)), DISPLAY_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
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


def plot_per_class_f1(predictions_path, output_path):
    df = pd.read_csv(predictions_path)
    methods = {
        "Audio only": "audio_pred",
        "Visual only": "visual_pred",
        "Fixed late fusion": "fusion_pred",
        "Learned late fusion": "learned_fusion_pred",
    }
    colours = ["#4477AA", "#EE7733", "#999999", "#2E8B57"]
    x = np.arange(len(LABELS))
    width = 0.19

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for index, ((name, column), colour) in enumerate(zip(methods.items(), colours)):
        _, _, f1, _ = precision_recall_fscore_support(
            df["target_class"], df[column], labels=LABELS, zero_division=0
        )
        bars = ax.bar(
            x + (index - 1.5) * width,
            f1 * 100,
            width,
            label=name,
            color=colour,
        )
        if name == "Learned late fusion":
            add_value_labels(bars, ax, fontsize=7)

    ax.set_ylabel("Per-class F1 score (%)")
    ax.set_xticks(x, DISPLAY_LABELS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "vggsound_learned_fusion"
    output_dir = base_dir / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ablation(
        input_dir / "learned_fusion_ablation_table.csv",
        output_dir / "learned_fusion_ablation.png",
    )
    plot_confusion(
        input_dir / "learned_fusion_test_predictions.csv",
        output_dir / "learned_fusion_confusion_matrix_normalized.png",
    )
    plot_per_class_f1(
        input_dir / "learned_fusion_test_predictions.csv",
        output_dir / "learned_fusion_per_class_f1_comparison.png",
    )

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
