import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run_eval(args, alpha):
    output_dir = Path(args.output_prefix + str(alpha))

    if output_dir.exists() and not args.force:
        ablation_file = output_dir / "vggsound_ablation_table.csv"
        predictions_file = output_dir / "vggsound_fusion_predictions.csv"
        if ablation_file.exists() and predictions_file.exists():
            print(f"Skipping alpha={alpha}; output already exists.")
            return output_dir

    cmd = [
        "python",
        args.eval_script,
        "--manifest_csv", args.manifest_csv,
        "--mapping_csv", args.mapping_csv,
        "--checkpoint", args.checkpoint,
        "--yolo_standard_model", args.yolo_standard_model,
        "--output_dir", str(output_dir),
        "--split", args.split,
        "--samples_per_class", str(args.samples_per_class),
        "--frame_skip", str(args.frame_skip),
        "--alpha", str(alpha),
        "--device", args.device,
    ]

    if args.core_only:
        cmd.append("--core_only")

    print(f"\nRunning alpha={alpha}")
    subprocess.run(cmd, check=True)
    return output_dir


def collect_ablation_results(output_dirs, summary_csv):
    rows = []

    for output_dir in output_dirs:
        path = output_dir / "vggsound_ablation_table.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        df["run"] = output_dir.name
        rows.append(df)

    if not rows:
        raise RuntimeError("No ablation tables found.")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(summary_csv, index=False)

    fusion = out[out["variant"].str.contains("late_fusion", na=False)].copy()
    fusion = fusion.sort_values("macro_f1", ascending=False)

    print("\nAll late-fusion alpha results")
    print(fusion[["run", "variant", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].to_string(index=False))

    best = fusion.iloc[0]
    print("\nBest alpha run")
    print(best[["run", "variant", "accuracy", "macro_precision", "macro_recall", "macro_f1"]].to_string())

    return out, best


def write_confusion_report(best_run, output_path):
    predictions_path = Path(best_run) / "vggsound_fusion_predictions.csv"

    if not predictions_path.exists():
        raise RuntimeError(f"Missing predictions file: {predictions_path}")

    df = pd.read_csv(predictions_path)

    lines = []

    lines.append("Class counts")
    lines.append(str(df["target_class"].value_counts()))
    lines.append("")

    lines.append("Audio confusion")
    lines.append(str(pd.crosstab(df["target_class"], df["audio_pred"])))
    lines.append("")

    lines.append("Visual confusion")
    lines.append(str(pd.crosstab(df["target_class"], df["visual_pred"])))
    lines.append("")

    lines.append("Fusion confusion")
    lines.append(str(pd.crosstab(df["target_class"], df["fusion_pred"])))
    lines.append("")

    output_path.write_text("\n".join(lines))

    print(f"\nSaved confusion report to: {output_path}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_script", default="vggsound_fusion_eval.py")
    parser.add_argument("--manifest_csv", default="../../datasets/vggsound/vggsound_subset_manifest.csv")
    parser.add_argument("--mapping_csv", default="vggsound_urban_map.csv")
    parser.add_argument("--checkpoint", default="../cnn-bilstm-audio/best_cnn_bilstm_large_v2.pt")
    parser.add_argument("--yolo_standard_model", default="yolov8n.pt")
    parser.add_argument("--output_prefix", default="vggsound_fusion_alpha_")
    parser.add_argument("--summary_csv", default="vggsound_alpha_sweep_summary.csv")
    parser.add_argument("--confusion_report", default="vggsound_best_confusion_report.txt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples_per_class", type=int, default=50)
    parser.add_argument("--frame_skip", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--core_only", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    output_dirs = []

    for alpha in args.alphas:
        output_dirs.append(run_eval(args, alpha))

    _, best = collect_ablation_results(output_dirs, args.summary_csv)

    best_run = best["run"]
    write_confusion_report(best_run, Path(args.confusion_report))

    print(f"\nSaved alpha sweep summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()