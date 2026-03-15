from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Input and output locations for the Milestone 3 modeling stage.
FEATURED_DATA_PATH = Path("data/processed/netflix_titles_featured.csv")
OUTPUT_DIR = Path("outputs/milestone3")


def load_dataset(path: Path = FEATURED_DATA_PATH) -> pd.DataFrame:
    """Load the featured Netflix dataset created in Milestone 2."""
    # Fail early when the expected Milestone 2 output is missing.
    if not path.exists():
        raise FileNotFoundError(f"Featured dataset not found: {path}")
    # Load the CSV so downstream modeling steps can reuse one shared source.
    return pd.read_csv(path)


def split_pipe_values(value: str | float | None) -> List[str]:
    """Split deterministic pipe-separated values into a clean token list."""
    # Convert missing or placeholder-like values into an empty token list.
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    # Split the serialized text, trim whitespace, and discard explicit unknowns.
    tokens = [token.strip() for token in str(value).split("|")]
    return [token for token in tokens if token and token != "Unknown"]


def slugify_label(label: str) -> str:
    """Convert a human-readable label into a stable feature-safe slug."""
    # Lowercase the label and replace non-alphanumeric runs with underscores.
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    # Preserve a fallback slug so feature names always remain valid.
    return slug or "unknown"


def get_top_labels(series: pd.Series, top_n: int) -> List[str]:
    """Return the most frequent labels from a pipe-separated categorical series."""
    # Count every token occurrence across all rows in the provided series.
    counts: Dict[str, int] = {}
    for value in series.fillna("Unknown"):
        for token in split_pipe_values(value):
            counts[token] = counts.get(token, 0) + 1

    # Sort by frequency and label for reproducible feature selection.
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [label for label, _ in ranked[:top_n]]


def add_indicator_columns(
    df: pd.DataFrame,
    source_column: str,
    labels: Iterable[str],
    prefix: str,
) -> pd.DataFrame:
    """Add one binary indicator column per requested multi-label token."""
    # Work on a copy so helper usage never mutates shared caller state.
    enriched = df.copy()
    label_list = list(labels)

    # Pre-split token lists once so each indicator column can reuse them cheaply.
    token_lists = enriched[source_column].fillna("Unknown").apply(split_pipe_values)
    for label in label_list:
        slug = slugify_label(label)
        enriched[f"{prefix}__{slug}"] = token_lists.apply(lambda values: int(label in values))

    # Return the enriched frame with deterministic indicator feature names.
    return enriched


def build_modeling_dataset(
    df: pd.DataFrame,
    *,
    top_genres: int = 12,
    top_countries: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Create a model-ready dataset and metadata for Milestone 3 tasks."""
    # Copy the source data so feature engineering remains side-effect free.
    modeling_df = df.copy()

    # Parse added dates once to derive year-based temporal modeling features.
    added_dates = pd.to_datetime(modeling_df["date_added"], errors="coerce")
    modeling_df["added_year"] = added_dates.dt.year.fillna(modeling_df["release_year"])

    # Fill duration numerics and track missingness so models retain that signal.
    modeling_df["duration_missing_flag"] = modeling_df["duration_value"].isna().astype(int)
    duration_median = float(modeling_df["duration_value"].median())
    modeling_df["duration_value"] = modeling_df["duration_value"].fillna(duration_median)

    # Add compact count features summarizing country and genre breadth per title.
    modeling_df["genre_count"] = modeling_df["listed_in"].fillna("Unknown").apply(
        lambda value: len(split_pipe_values(value))
    )
    modeling_df["country_count"] = modeling_df["country"].fillna("Unknown").apply(
        lambda value: len(split_pipe_values(value))
    )

    # Measure how long after release each title appeared on Netflix.
    modeling_df["years_to_platform"] = (
        modeling_df["added_year"].astype(float) - modeling_df["release_year"].astype(float)
    )
    modeling_df["years_to_platform"] = modeling_df["years_to_platform"].clip(lower=0).fillna(0)

    # Select the most common multi-label categories for compact indicator features.
    top_genre_labels = get_top_labels(modeling_df["listed_in"], top_genres)
    top_country_labels = get_top_labels(modeling_df["country"], top_countries)

    # Add reusable genre and country indicator columns for downstream analyses.
    modeling_df = add_indicator_columns(modeling_df, "listed_in", top_genre_labels, "genre")
    modeling_df = add_indicator_columns(modeling_df, "country", top_country_labels, "country")

    # Return both the engineered frame and the selected label metadata.
    metadata = {
        "top_genres": top_genre_labels,
        "top_countries": top_country_labels,
    }
    return modeling_df, metadata


def encode_feature_matrix(
    df: pd.DataFrame,
    *,
    numeric_columns: List[str],
    categorical_columns: List[str],
    indicator_prefixes: List[str],
) -> pd.DataFrame:
    """Convert selected columns into a numeric feature matrix for modeling."""
    # Collect binary indicator columns that match the requested prefixes.
    indicator_columns = [
        column
        for column in df.columns
        if any(column.startswith(f"{prefix}__") for prefix in indicator_prefixes)
    ]

    # Combine numeric, categorical, and binary indicator inputs in one frame.
    feature_frame = df[numeric_columns + categorical_columns + indicator_columns].copy()

    # One-hot encode compact categorical fields while preserving feature names.
    encoded = pd.get_dummies(feature_frame, columns=categorical_columns, dtype=int)
    return encoded


def compute_top_indicator_labels(
    df: pd.DataFrame,
    *,
    cluster_column: str,
    indicator_prefix: str,
    labels: List[str],
    top_n: int = 3,
) -> pd.Series:
    """Return the dominant labels inside each cluster from indicator columns."""
    # Build a deterministic mapping from labels to their indicator column names.
    label_map = {label: f"{indicator_prefix}__{slugify_label(label)}" for label in labels}

    # Rank label prevalence per cluster and keep the top few for reporting.
    summaries = {}
    grouped = df.groupby(cluster_column)
    for cluster_id, cluster_df in grouped:
        ranked = sorted(
            (
                (label, float(cluster_df[column].mean()))
                for label, column in label_map.items()
                if column in cluster_df.columns
            ),
            key=lambda item: (-item[1], item[0]),
        )
        top_labels = [f"{label} ({share:.0%})" for label, share in ranked[:top_n]]
        summaries[cluster_id] = ", ".join(top_labels)

    # Return one descriptive string per cluster for tables and notebook displays.
    return pd.Series(summaries)


def run_clustering(
    modeling_df: pd.DataFrame,
    metadata: Dict[str, List[str]],
) -> Dict[str, object]:
    """Cluster Netflix titles by genre, duration, and ratings."""
    # Build the clustering feature matrix around the Milestone 3 requirement fields.
    feature_matrix = encode_feature_matrix(
        modeling_df,
        numeric_columns=["duration_value", "genre_count", "years_to_platform"],
        categorical_columns=["rating", "duration_unit", "length_category"],
        indicator_prefixes=["genre"],
    )

    # Standardize the mixed feature matrix so KMeans treats inputs comparably.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # Search a small cluster range and score each option with silhouette score.
    silhouette_rows = []
    best_k = 2
    best_score = -1.0
    for candidate_k in range(2, 7):
        model = KMeans(n_clusters=candidate_k, n_init=20, random_state=42)
        labels = model.fit_predict(scaled_features)
        score = float(silhouette_score(scaled_features, labels))
        silhouette_rows.append({"n_clusters": candidate_k, "silhouette_score": score})
        if score > best_score:
            best_k = candidate_k
            best_score = score

    # Fit the final clustering model with the best-performing cluster count.
    final_model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    cluster_labels = final_model.fit_predict(scaled_features)
    clustered_df = modeling_df.copy()
    clustered_df["cluster"] = cluster_labels

    # Summarize each cluster using size, content mix, and dominant label patterns.
    cluster_profile = (
        clustered_df.groupby("cluster")
        .agg(
            titles=("show_id", "count"),
            movie_share=("type", lambda values: float((values == "Movie").mean())),
            avg_duration=("duration_value", "mean"),
            avg_release_year=("release_year", "mean"),
            avg_years_to_platform=("years_to_platform", "mean"),
        )
        .reset_index()
    )
    cluster_profile["top_genres"] = cluster_profile["cluster"].map(
        compute_top_indicator_labels(
            clustered_df,
            cluster_column="cluster",
            indicator_prefix="genre",
            labels=metadata["top_genres"],
            top_n=3,
        )
    )

    # Capture the most frequent rating label inside each cluster for interpretation.
    top_ratings = (
        clustered_df.groupby("cluster")["rating"]
        .agg(lambda values: values.value_counts().idxmax())
        .rename("top_rating")
    )
    cluster_profile["top_rating"] = cluster_profile["cluster"].map(top_ratings)

    # Return both the row-level assignments and the cluster-level summary outputs.
    return {
        "best_k": best_k,
        "best_score": best_score,
        "silhouette_scores": pd.DataFrame(silhouette_rows),
        "clustered_df": clustered_df,
        "cluster_profile": cluster_profile.sort_values("titles", ascending=False),
    }


def format_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    *,
    top_n: int = 15,
) -> pd.DataFrame:
    """Format raw importance vectors into a readable ranked DataFrame."""
    # Pair each importance value with its feature name and keep the strongest rows.
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    return importance_df.sort_values("importance", ascending=False).head(top_n).reset_index(
        drop=True
    )


def run_type_classification(modeling_df: pd.DataFrame) -> Dict[str, object]:
    """Classify content type as Movie vs. TV Show from engineered features."""
    # Build a rich feature matrix using temporal, categorical, and label indicators.
    feature_matrix = encode_feature_matrix(
        modeling_df,
        numeric_columns=[
            "release_year",
            "added_year",
            "duration_value",
            "genre_count",
            "country_count",
            "years_to_platform",
            "duration_missing_flag",
        ],
        categorical_columns=["rating", "duration_unit", "length_category", "content_origin"],
        indicator_prefixes=["genre", "country"],
    )
    target = modeling_df["type"].copy()

    # Reserve a stratified holdout set so evaluation reflects unseen data.
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    # Fit a random forest that can handle the mixed feature space robustly.
    classifier = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    classifier.fit(X_train, y_train)

    # Score the holdout set and retain detailed metrics for notebook reporting.
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    confusion = confusion_matrix(y_test, predictions, labels=["Movie", "TV Show"])
    accuracy = float(accuracy_score(y_test, predictions))
    weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))

    # Estimate importance on the holdout set to reflect real predictive usefulness.
    permutation = permutation_importance(
        classifier,
        X_test,
        y_test,
        n_repeats=8,
        random_state=42,
        scoring="f1_weighted",
        n_jobs=1,
    )
    top_features = format_feature_importances(
        X_test.columns.tolist(), permutation.importances_mean, top_n=15
    )

    # Return evaluation metrics, confusion matrix, and ranked feature drivers.
    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": pd.DataFrame(
            confusion,
            index=["actual_movie", "actual_tv_show"],
            columns=["pred_movie", "pred_tv_show"],
        ),
        "top_features": top_features,
        "test_size": int(len(X_test)),
        "class_probabilities_shape": list(probabilities.shape),
    }


def run_binary_driver_models(
    modeling_df: pd.DataFrame,
    *,
    target_labels: List[str],
    target_prefix: str,
    numeric_columns: List[str],
    categorical_columns: List[str],
    indicator_prefixes: List[str],
) -> Dict[str, pd.DataFrame]:
    """Train one-vs-rest models to identify drivers for top countries or genres."""
    # Build one shared feature matrix for the full family of driver models.
    feature_matrix = encode_feature_matrix(
        modeling_df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        indicator_prefixes=indicator_prefixes,
    )

    # Keep per-target model quality metrics and per-feature importance summaries.
    performance_rows = []
    importance_rows = []

    # Train one binary classifier per label to avoid forcing arbitrary single labels.
    for label in target_labels:
        target_column = f"{target_prefix}__{slugify_label(label)}"
        if target_column not in modeling_df.columns:
            continue

        # Skip labels that do not have enough positive examples for a stable split.
        target = modeling_df[target_column].astype(int)
        positive_count = int(target.sum())
        negative_count = int((1 - target).sum())
        if positive_count < 80 or negative_count < 80:
            continue

        # Split each one-vs-rest task with stratification to preserve class balance.
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix,
            target,
            test_size=0.2,
            random_state=42,
            stratify=target,
        )

        # Fit a balanced random forest suited to mixed sparse tabular signals.
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        )
        classifier.fit(X_train, y_train)

        # Score each target and keep the metrics compact for later comparison.
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)[:, 1]
        performance_rows.append(
            {
                "label": label,
                "positives": positive_count,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "f1": float(f1_score(y_test, predictions)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
        )

        # Measure which inputs most improve holdout-set F1 for the current label.
        permutation = permutation_importance(
            classifier,
            X_test,
            y_test,
            n_repeats=6,
            random_state=42,
            scoring="f1",
            n_jobs=1,
        )
        for feature, importance in zip(X_test.columns, permutation.importances_mean):
            importance_rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    # Aggregate importances across targets to surface the broadest common drivers.
    performance_df = pd.DataFrame(performance_rows).sort_values("f1", ascending=False)
    importance_df = pd.DataFrame(importance_rows)
    average_importance_df = (
        importance_df.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(15)
    )

    # Return both per-label performance and shared feature-importance summaries.
    return {
        "performance": performance_df.reset_index(drop=True),
        "importance_by_label": importance_df.sort_values(
            ["label", "importance"], ascending=[True, False]
        ).reset_index(drop=True),
        "average_importance": average_importance_df.reset_index(drop=True),
    }


def run_availability_driver_analysis(
    modeling_df: pd.DataFrame,
    metadata: Dict[str, List[str]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Analyze what features drive title availability across top countries and genres.

    Because both countries and genres are multi-label fields, this analysis treats
    availability as a family of one-vs-rest membership tasks rather than forcing a
    single arbitrary label per title.
    """
    # Model title membership in the most common countries using genre-rich features.
    country_results = run_binary_driver_models(
        modeling_df,
        target_labels=metadata["top_countries"][:6],
        target_prefix="country",
        numeric_columns=[
            "release_year",
            "added_year",
            "duration_value",
            "genre_count",
            "years_to_platform",
            "duration_missing_flag",
        ],
        categorical_columns=["type", "rating", "duration_unit", "length_category", "content_origin"],
        indicator_prefixes=["genre"],
    )

    # Model title membership in the most common genres using country-rich features.
    genre_results = run_binary_driver_models(
        modeling_df,
        target_labels=metadata["top_genres"][:6],
        target_prefix="genre",
        numeric_columns=[
            "release_year",
            "added_year",
            "duration_value",
            "country_count",
            "years_to_platform",
            "duration_missing_flag",
        ],
        categorical_columns=["type", "rating", "duration_unit", "length_category", "content_origin"],
        indicator_prefixes=["country"],
    )

    # Return separate country and genre driver analyses for notebook storytelling.
    return {
        "country": country_results,
        "genre": genre_results,
    }


def save_results(
    clustering_results: Dict[str, object],
    classification_results: Dict[str, object],
    driver_results: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    """Persist Milestone 3 result tables for reuse outside the notebook."""
    # Ensure the shared output directory exists before writing any artifacts.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cluster-level and row-level outputs for downstream inspection.
    clustering_results["silhouette_scores"].to_csv(
        OUTPUT_DIR / "cluster_silhouette_scores.csv", index=False
    )
    clustering_results["cluster_profile"].to_csv(OUTPUT_DIR / "cluster_profile.csv", index=False)
    clustering_results["clustered_df"][
        ["show_id", "title", "type", "rating", "duration", "listed_in", "cluster"]
    ].to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)

    # Save classification metrics and the strongest feature-importance signals.
    classification_results["confusion_matrix"].to_csv(
        OUTPUT_DIR / "type_classification_confusion_matrix.csv"
    )
    classification_results["top_features"].to_csv(
        OUTPUT_DIR / "type_classification_top_features.csv", index=False
    )

    # Persist country and genre driver tables for easy review in git or Excel.
    driver_results["country"]["performance"].to_csv(
        OUTPUT_DIR / "country_driver_performance.csv", index=False
    )
    driver_results["country"]["average_importance"].to_csv(
        OUTPUT_DIR / "country_driver_average_importance.csv", index=False
    )
    driver_results["genre"]["performance"].to_csv(
        OUTPUT_DIR / "genre_driver_performance.csv", index=False
    )
    driver_results["genre"]["average_importance"].to_csv(
        OUTPUT_DIR / "genre_driver_average_importance.csv", index=False
    )

    # Write a compact JSON summary for quick machine-readable milestone checks.
    summary = {
        "best_cluster_count": int(clustering_results["best_k"]),
        "best_silhouette_score": float(clustering_results["best_score"]),
        "type_classification_accuracy": float(classification_results["accuracy"]),
        "type_classification_weighted_f1": float(classification_results["weighted_f1"]),
        "country_driver_labels": driver_results["country"]["performance"]["label"].tolist(),
        "genre_driver_labels": driver_results["genre"]["performance"]["label"].tolist(),
    }
    with (OUTPUT_DIR / "milestone3_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    """Run the full Milestone 3 modeling workflow."""
    # Execute the shared modeling workflow from dataset load to artifact save.
    featured_df = load_dataset()
    modeling_df, metadata = build_modeling_dataset(featured_df)
    clustering_results = run_clustering(modeling_df, metadata)
    classification_results = run_type_classification(modeling_df)
    driver_results = run_availability_driver_analysis(modeling_df, metadata)
    save_results(clustering_results, classification_results, driver_results)

    # Print a concise summary so command-line runs remain easy to verify.
    print("Milestone 3 modeling complete.")
    print(f"Best cluster count: {clustering_results['best_k']}")
    print(f"Best silhouette score: {clustering_results['best_score']:.4f}")
    print(f"Type classification accuracy: {classification_results['accuracy']:.4f}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Allow running this module directly as a script.
    main()
