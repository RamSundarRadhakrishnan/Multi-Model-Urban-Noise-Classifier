from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


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


def main():
    input_path = Path(
        "vggsound_fusion_alpha_0.5/vggsound_fusion_predictions.csv"
    )
    output_dir = Path("paper_figures")
    output_path = output_dir / "vggsound_per_class_f1_by_modality.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    prediction_columns = {
        "Audio only": "audio_pred",
        "Visual only": "visual_pred",
        "Late fusion": "fusion_pred",
    }
    colors = ["#4477AA", "#EE7733", "#2E8B57"]

    x = np.arange(len(LABELS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    for index, ((name, column), color) in enumerate(
        zip(prediction_columns.items(), colors)
    ):
        _, _, f1, _ = precision_recall_fscore_support(
            df["target_class"],
            df[column],
            labels=LABELS,
            zero_division=0,
        )

        bars = ax.bar(
            x + (index - 1) * width,
            f1 * 100,
            width,
            label=name,
            color=color,
        )

        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_ylabel("Per-class F1 score (%)")
    ax.set_xticks(x, DISPLAY_LABELS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()