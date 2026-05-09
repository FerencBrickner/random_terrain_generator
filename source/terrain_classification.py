"""
Train a PRNG classifier from terrain statistics and show which artifacts each PRNG tends to produce.
IDEA: https://pypi.org/project/scikit-learn/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from db import Session, TerrainStats


OUTPUT_DIRECTORY: Final[Path] = Path("terrain_prng_classification_output")


@dataclass
class PreparedDataset:
    feature_table: pd.DataFrame
    label_series: pd.Series
    feature_column_names: List[str]


def load_terrain_rows() -> pd.DataFrame:
    with Session() as database_session:
        terrain_rows = database_session.query(TerrainStats).all()

    if not terrain_rows:
        raise ValueError("No rows found in terrain_stats.")

    return pd.DataFrame(
        {
            "prng_type": terrain_row.prng_type,
            "total_values": terrain_row.total_values,
            "min": terrain_row.min,
            "max": terrain_row.max,
            "mean": terrain_row.mean,
            "median": terrain_row.median,
            "std": terrain_row.std,
            "variance": terrain_row.variance,
            "range": terrain_row.range,
            "percentile25": terrain_row.percentile25,
            "percentile75": terrain_row.percentile75,
            "interquartile_range": terrain_row.interquartile_range,
            "skewness": terrain_row.skewness,
            "kurtosis": terrain_row.kurtosis,
            "unique_values": terrain_row.unique_values,
            "mean_gradient": terrain_row.mean_gradient,
            "max_gradient": terrain_row.max_gradient,
            "histogram_counts": terrain_row.histogram_counts,
            "histogram_bins": terrain_row.histogram_bins,
            "morans_i_spatial_correlation": terrain_row.morans_i_spatial_correlation,
            "gearys_c_spatial_correlation": terrain_row.gearys_c_spatial_correlation
        }
        for terrain_row in terrain_rows
    )


def safe_json_load(value: Any) -> Optional[Any]:
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def expand_histogram_features(terrain_frame: pd.DataFrame) -> pd.DataFrame:
    expanded_terrain_frame = terrain_frame.copy()
    histogram_count_entropy_values: List[float] = []
    histogram_count_dominance_ratio_values: List[float] = []
    histogram_count_uniformity_score_values: List[float] = []
    histogram_count_peak_to_mean_ratio_values: List[float] = []

    for histogram_counts_json in expanded_terrain_frame["histogram_counts"]:
        histogram_counts = safe_json_load(histogram_counts_json)
        if histogram_counts is None:
            histogram_count_entropy_values.append(np.nan)
            histogram_count_dominance_ratio_values.append(np.nan)
            histogram_count_uniformity_score_values.append(np.nan)
            histogram_count_peak_to_mean_ratio_values.append(np.nan)
            continue

        histogram_count_array = np.asarray(histogram_counts, dtype=float)
        total_histogram_mass = float(histogram_count_array.sum())
        if histogram_count_array.size == 0 or total_histogram_mass <= 0:
            histogram_count_entropy_values.append(np.nan)
            histogram_count_dominance_ratio_values.append(np.nan)
            histogram_count_uniformity_score_values.append(np.nan)
            histogram_count_peak_to_mean_ratio_values.append(np.nan)
            continue

        histogram_probabilities = histogram_count_array / total_histogram_mass
        nonzero_probabilities = histogram_probabilities[histogram_probabilities > 0]
        histogram_entropy_value = float(-(nonzero_probabilities * np.log2(nonzero_probabilities)).sum())
        histogram_maximum_count = float(histogram_count_array.max())
        histogram_mean_count = float(histogram_count_array.mean())
        uniform_probability = 1.0 / len(histogram_probabilitys) if (histogram_probabilitys := histogram_probabilities).size else np.nan
        histogram_uniformity_score_value = float(1.0 - np.abs(histogram_probabilities - uniform_probability).sum() / 2.0)

        histogram_count_entropy_values.append(histogram_entropy_value)
        histogram_count_dominance_ratio_values.append(float(histogram_maximum_count / total_histogram_mass))
        histogram_count_uniformity_score_values.append(histogram_uniformity_score_value)
        histogram_count_peak_to_mean_ratio_values.append(float(histogram_maximum_count / histogram_mean_count))

    expanded_terrain_frame["histogram_count_entropy"] = histogram_count_entropy_values
    expanded_terrain_frame["histogram_count_dominance_ratio"] = histogram_count_dominance_ratio_values
    expanded_terrain_frame["histogram_count_uniformity_score"] = histogram_count_uniformity_score_values
    expanded_terrain_frame["histogram_count_peak_to_mean_ratio"] = histogram_count_peak_to_mean_ratio_values
    return expanded_terrain_frame


def prepare_dataset(terrain_frame: pd.DataFrame) -> PreparedDataset:
    expanded_terrain_frame = expand_histogram_features(terrain_frame)
    ignored_columns = {"prng_type", "histogram_counts", "histogram_bins"}
    feature_column_names = [column_name for column_name in expanded_terrain_frame.columns if column_name not in ignored_columns]
    return PreparedDataset(
        feature_table=expanded_terrain_frame[feature_column_names].copy(),
        label_series=expanded_terrain_frame["prng_type"].astype(str).copy(),
        feature_column_names=feature_column_names,
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                ),
            ),
        ]
    )


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series
    predicted_labels: np.ndarray
    evaluation_metrics: Dict[str, float]
    cross_validation_summary: Dict[str, float]


def train_model(feature_table: pd.DataFrame, label_series: pd.Series) -> TrainingResult:
    minimum_class_count = int(label_series.value_counts().min())
    if minimum_class_count < 2:
        raise ValueError("Each PRNG type needs at least two rows.")

    x_train, x_test, y_train, y_test = train_test_split(
        feature_table,
        label_series,
        test_size=0.2,
        random_state=42,
        stratify=label_series,
    )

    trained_pipeline = build_pipeline()
    trained_pipeline.fit(x_train, y_train)
    predicted_labels = trained_pipeline.predict(x_test)

    evaluation_metrics = {
        "accuracy": accuracy_score(y_test, predicted_labels),
        "macro_f1_score": f1_score(y_test, predicted_labels, average="macro"),
        "weighted_f1_score": f1_score(y_test, predicted_labels, average="weighted"),
    }

    cross_validation_folds = min(5, minimum_class_count)
    cross_validator = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=42)
    cross_validation_results = cross_validate(
        trained_pipeline,
        feature_table,
        label_series,
        cv=cross_validator,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
    )

    cross_validation_summary = {
        "mean_cross_validated_accuracy": float(np.mean(cross_validation_results["test_accuracy"])),
        "mean_cross_validated_macro_f1": float(np.mean(cross_validation_results["test_macro_f1"])),
    }

    return TrainingResult(
        trained_pipeline=trained_pipeline,
        x_test=x_test,
        y_test=y_test,
        predicted_labels=predicted_labels,
        evaluation_metrics=evaluation_metrics,
        cross_validation_summary=cross_validation_summary,
    )


def save_confusion_matrix_plot(y_true: Sequence[str], y_pred: Sequence[str], class_names: Sequence[str]) -> Path:
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=class_names), display_labels=class_names).plot(
        ax=axis,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=True,
    )
    axis.set_title("PRNG Classification Confusion Matrix")
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / "confusion_matrix.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_feature_importance_plot(trained_pipeline: Pipeline, feature_column_names: Sequence[str]) -> Path:
    feature_importances = trained_pipeline.named_steps["classifier"].feature_importances_
    feature_importance_frame = (
        pd.DataFrame({"feature_name": feature_column_names, "importance": feature_importances})
        .sort_values("importance", ascending=False)
        .head(15)
        .iloc[::-1]
    )

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(feature_importance_frame["feature_name"], feature_importance_frame["importance"])
    axis.set_title("Top Feature Importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / "feature_importances.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_permutation_importance_plot(trained_pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Path:
    permutation_importance_result = permutation_importance(
        trained_pipeline,
        x_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1_macro",
    )

    permutation_importance_frame = (
        pd.DataFrame(
            {
                "feature_name": list(x_test.columns),
                "mean_importance": permutation_importance_result.importances_mean,
                "importance_standard_deviation": permutation_importance_result.importances_std,
            }
        )
        .sort_values("mean_importance", ascending=False)
        .head(15)
        .iloc[::-1]
    )

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(
        permutation_importance_frame["feature_name"],
        permutation_importance_frame["mean_importance"],
        xerr=permutation_importance_frame["importance_standard_deviation"],
    )
    axis.set_title("Permutation Importance")
    axis.set_xlabel("Mean importance")
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / "permutation_importances.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def print_prng_artifact_leaderboard(terrain_frame: pd.DataFrame) -> None:
    expanded_terrain_frame = expand_histogram_features(terrain_frame)
    artifact_columns = [
        "kurtosis",
        "skewness",
        "std",
        "mean_gradient",
        "max_gradient",
        "unique_values",
        "interquartile_range",
        "histogram_count_entropy",
        "histogram_count_dominance_ratio",
        "histogram_count_uniformity_score",
        "histogram_count_peak_to_mean_ratio",
    ]

    print("Artifact leaders by PRNG")
    prng_artifact_medians = expanded_terrain_frame.groupby("prng_type")[artifact_columns].median(numeric_only=True)
    for artifact_column_name in artifact_columns:
        ranked_prngs = prng_artifact_medians[artifact_column_name].sort_values(ascending=False)
        print(f"{artifact_column_name}: {ranked_prngs.index[0]} = {ranked_prngs.iloc[0]:.4f}")
        print("  top 3: " + ", ".join(f"{prng_name} ({value:.4f})" for prng_name, value in ranked_prngs.head(3).items()))


def save_prng_artifact_plots(terrain_frame: pd.DataFrame) -> List[Path]:
    expanded_terrain_frame = expand_histogram_features(terrain_frame)
    artifact_columns = [
        "kurtosis",
        "skewness",
        "std",
        "mean_gradient",
        "max_gradient",
        "unique_values",
        "interquartile_range",
        "histogram_count_entropy",
        "histogram_count_dominance_ratio",
        "histogram_count_uniformity_score",
        "histogram_count_peak_to_mean_ratio",
    ]

    prng_artifact_medians = expanded_terrain_frame.groupby("prng_type")[artifact_columns].median(numeric_only=True)
    plot_paths: List[Path] = []

    for artifact_column_name in artifact_columns:
        figure, axis = plt.subplots(figsize=(12, 6))
        prng_artifact_medians[artifact_column_name].sort_values(ascending=False).plot(kind="bar", ax=axis)
        axis.set_title(f"Median {artifact_column_name} by PRNG")
        axis.set_xlabel("PRNG type")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        plot_path = OUTPUT_DIRECTORY / f"median_{artifact_column_name}_by_prng.png"
        figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(plot_path)

    correlation_frame = expanded_terrain_frame[artifact_columns].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    heatmap = axis.imshow(correlation_frame.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(artifact_columns)))
    axis.set_yticks(np.arange(len(artifact_columns)))
    axis.set_xticklabels(artifact_columns, rotation=45, ha="right")
    axis.set_yticklabels(artifact_columns)
    axis.set_title("Artifact Correlation Matrix")
    figure.colorbar(heatmap, ax=axis, shrink=0.8)
    figure.tight_layout()
    plot_path = OUTPUT_DIRECTORY / "artifact_correlation_matrix.png"
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(plot_path)

    return plot_paths


def print_summary_report(terrain_frame: pd.DataFrame, training_result: TrainingResult, plot_paths: Sequence[Path]) -> None:
    print("Dataset summary")
    print(f"Rows loaded: {len(terrain_frame)}")
    print("Class counts:")
    print(terrain_frame["prng_type"].value_counts().to_string())

    print("Classification metrics")
    for metric_name, metric_value in training_result.evaluation_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Cross-validation summary")
    for metric_name, metric_value in training_result.cross_validation_summary.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("Per-class report")
    print(classification_report(training_result.y_test, training_result.predicted_labels, zero_division=0))

    print("Saved plots")
    for plot_path in plot_paths:
        print(plot_path)


def main() -> None:
    terrain_frame = load_terrain_rows()
    prepared_dataset = prepare_dataset(terrain_frame)
    training_result = train_model(prepared_dataset.feature_table, prepared_dataset.label_series)

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    confusion_matrix_path = save_confusion_matrix_plot(
        training_result.y_test,
        training_result.predicted_labels,
        sorted(training_result.y_test.unique()),
    )
    feature_importance_path = save_feature_importance_plot(training_result.trained_pipeline, prepared_dataset.feature_column_names)
    permutation_importance_path = save_permutation_importance_plot(
        training_result.trained_pipeline,
        training_result.x_test,
        training_result.y_test,
    )
    artifact_plot_paths = save_prng_artifact_plots(terrain_frame)

    print_prng_artifact_leaderboard(terrain_frame)
    print_summary_report(
        terrain_frame,
        training_result,
        [confusion_matrix_path, feature_importance_path, permutation_importance_path, *artifact_plot_paths],
    )


if __name__ == "__main__":
    main()
