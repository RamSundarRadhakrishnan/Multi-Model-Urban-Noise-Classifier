"""Train a learned late-fusion head from frozen audio and visual VGGSound scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CLASSES = [
    "construction-site",
    "crowd-noise",
    "motorvehicle-horn",
    "motorvehicle-siren",
]


def parse_scores(value: object, classes: list[str]) -> list[float]:
    if isinstance(value, str):
        values = json.loads(value)
    elif isinstance(value, dict):
        values = value
    else:
        values = {}
    return [float(values.get(name, 0.0)) for name in classes]


def make_features(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    required = {"audio_scores_json", "visual_scores_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")

    audio = np.asarray([parse_scores(value, classes) for value in df["audio_scores_json"]])
    visual = np.asarray([parse_scores(value, classes) for value in df["visual_scores_json"]])
    return np.concatenate([audio, visual], axis=1)


def metrics(y_true: pd.Series, y_pred: np.ndarray, classes: list[str]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def get_classes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    observed = set(train_df["target_class"]) | set(val_df["target_class"]) | set(test_df["target_class"])
    if set(DEFAULT_CLASSES).issubset(observed):
        return DEFAULT_CLASSES
    return sorted(observed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_predictions", required=True)
    parser.add_argument("--val_predictions", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--output_dir", default="vggsound_learned_fusion")
    parser.add_argument("--c_values", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--max_iter", type=int, default=5000)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_predictions)
    val_df = pd.read_csv(args.val_predictions)
    test_df = pd.read_csv(args.test_predictions)
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if "target_class" not in df.columns:
            raise ValueError(f"{name} predictions do not contain target_class")

    classes = get_classes(train_df, val_df, test_df)
    x_train, y_train = make_features(train_df, classes), train_df["target_class"]
    x_val, y_val = make_features(val_df, classes), val_df["target_class"]
    x_test, y_test = make_features(test_df, classes), test_df["target_class"]

    selection_rows = []
    best_model = None
    best_c = None
    best_key = None
    for c_value in args.c_values:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=args.max_iter,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        result = metrics(y_val, model.predict(x_val), classes)
        selection_rows.append({"C": c_value, **result})
        key = (result["macro_f1"], result["accuracy"])
        if best_key is None or key > best_key:
            best_key, best_model, best_c = key, model, c_value

    # Refit the selected fixed configuration on all non-test score-label pairs.
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)
    final_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=best_c,
                    max_iter=args.max_iter,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    final_model.fit(x_fit, y_fit)

    learned_pred = final_model.predict(x_test)
    learned_prob = final_model.predict_proba(x_test)
    output = test_df.copy()
    output["learned_fusion_pred"] = learned_pred
    output["learned_fusion_conf"] = learned_prob.max(axis=1)
    output["learned_fusion_correct"] = learned_pred == y_test.to_numpy()

    rows = []
    for name, column in [
        ("audio_only", "audio_pred"),
        ("visual_only", "visual_pred"),
        ("fixed_equal_weight_fusion", "fusion_pred"),
    ]:
        if column in test_df.columns:
            rows.append({"variant": name, **metrics(y_test, test_df[column], classes)})
    rows.append({"variant": "learned_logistic_regression_fusion", **metrics(y_test, learned_pred, classes)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "learned_fusion_test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "learned_fusion_ablation_table.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "fusion_validation_model_selection.csv", index=False)
    joblib.dump(final_model, output_dir / "learned_fusion_logistic_regression.joblib")

    report = {
        "classes": classes,
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "selected_regularisation_C": best_c,
        "validation_selection_results": selection_rows,
        "test_ablation": rows,
        "learned_fusion_classification_report": classification_report(
            y_test, learned_pred, labels=classes, output_dict=True, zero_division=0
        ),
        "learned_fusion_confusion_matrix": confusion_matrix(
            y_test, learned_pred, labels=classes
        ).tolist(),
    }
    with open(output_dir / "learned_fusion_report.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Validation model selection:")
    print(pd.DataFrame(selection_rows).to_string(index=False))
    print("\nLocked-test ablation:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
