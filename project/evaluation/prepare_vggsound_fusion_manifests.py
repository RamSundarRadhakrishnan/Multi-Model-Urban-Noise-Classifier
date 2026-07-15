import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def resolve_video_path(value, manifest_dir):
    path = Path(str(value))

    if path.exists():
        return str(path.resolve())

    candidate = manifest_dir / path
    if candidate.exists():
        return str(candidate.resolve())

    return None


def sample_per_class(df, limit, seed):
    if limit is None or limit <= 0:
        return df.reset_index(drop=True)

    return (
        df.groupby("target_class", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), limit), random_state=seed))
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_manifest",
        required=True,
        help="Manifest created by vggsound_downloader.py, e.g. vggsound_train_manifest.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="../../datasets/vggsound/fusion_manifests",
    )
    parser.add_argument("--train_per_class", type=int, default=0)
    parser.add_argument("--val_per_class", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1.")

    manifest_path = Path(args.input_manifest).resolve()
    df = pd.read_csv(manifest_path)

    required = {
        "clip_id",
        "youtube_id",
        "start_seconds",
        "vggsound_label",
        "target_class",
        "video_path",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")

    df = df.copy()

    df["video_path"] = df["video_path"].map(
        lambda value: resolve_video_path(value, manifest_path.parent)
    )

    df = df.dropna(subset=["video_path"])

    if df.empty:
        raise RuntimeError("No existing video paths were found in the downloaded manifest.")

    train_parts = []
    val_parts = []

    for target_class, group in df.groupby("target_class"):
        if group["youtube_id"].nunique() < 2:
            raise RuntimeError(
                f"Not enough distinct YouTube IDs for class: {target_class}"
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.val_ratio,
            random_state=args.seed,
        )

        train_idx, val_idx = next(
            splitter.split(group, groups=group["youtube_id"])
        )

        train_parts.append(group.iloc[train_idx])
        val_parts.append(group.iloc[val_idx])

    train_df = sample_per_class(
        pd.concat(train_parts),
        args.train_per_class,
        args.seed,
    )

    val_df = sample_per_class(
        pd.concat(val_parts),
        args.val_per_class,
        args.seed,
    )

    train_df["split"] = "fusion_train"
    val_df["split"] = "fusion_val"

    columns = [
        "clip_id",
        "youtube_id",
        "start_seconds",
        "vggsound_label",
        "target_class",
        "split",
        "video_path",
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.loc[:, columns].to_csv(
        output_dir / "vggsound_fusion_train_manifest.csv",
        index=False,
    )

    val_df.loc[:, columns].to_csv(
        output_dir / "vggsound_fusion_val_manifest.csv",
        index=False,
    )

    summary = pd.concat([train_df, val_df]).groupby(
        ["split", "target_class"]
    ).size()

    print("Prepared fusion manifests from downloaded clips:")
    print(summary.to_string())
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()