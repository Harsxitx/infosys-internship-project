from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

# Output location for the generated Milestone 3 notebook.
NOTEBOOK_PATH = Path("notebooks/03_milestone3_modeling.ipynb")


def dedent_cell(source: str) -> str:
    """Normalize indentation for cleaner notebook cell source."""
    # Strip shared leading whitespace while preserving cell formatting.
    return textwrap.dedent(source).strip() + "\n"


def build_notebook() -> nbf.NotebookNode:
    """Create the Milestone 3 notebook structure and cell content."""
    # Initialize a fresh notebook container for the modeling milestone.
    notebook = nbf.v4.new_notebook()
    cells = []

    # Add a title cell that frames the milestone scope and deliverables.
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent_cell(
                """
                # Milestone 3: Modeling & Advanced Analysis

                This notebook covers:
                - Clustering Netflix titles by genre, duration, and rating patterns
                - Classification of content type (`Movie` vs. `TV Show`)
                - Driver analysis for content availability across top countries and genres
                - Feature-importance interpretation for each supervised modeling task
                """
            )
        )
    )

    # Add imports and notebook display configuration in one reusable setup cell.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Import analysis libraries, configure notebook display, and expose the src package.
                import sys
                from pathlib import Path

                import matplotlib.pyplot as plt
                import pandas as pd
                import seaborn as sns

                PROJECT_ROOT = Path.cwd().resolve()
                if not (PROJECT_ROOT / "src").exists():
                    PROJECT_ROOT = PROJECT_ROOT.parent
                if str(PROJECT_ROOT / "src") not in sys.path:
                    sys.path.append(str(PROJECT_ROOT / "src"))

                from modeling import (
                    build_modeling_dataset,
                    load_dataset,
                    run_availability_driver_analysis,
                    run_clustering,
                    run_type_classification,
                    save_results,
                )

                pd.set_option("display.max_columns", 100)
                pd.set_option("display.width", 140)
                sns.set_theme(style="whitegrid", palette="deep")
                """
            )
        )
    )

    # Load the featured dataset and run the shared modeling preparation helper.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Load the Milestone 2 dataset, prepare modeling features, and preview the result.
                featured_df = load_dataset()
                modeling_df, metadata = build_modeling_dataset(featured_df)

                print(f"Featured dataset shape: {featured_df.shape}")
                print(f"Modeling dataset shape: {modeling_df.shape}")
                print(f"Top genre labels: {metadata['top_genres']}")
                print(f"Top country labels: {metadata['top_countries']}")

                modeling_df.head()
                """
            )
        )
    )

    # Introduce the clustering section before visual and tabular outputs.
    cells.append(nbf.v4.new_markdown_cell("## Clustering Titles"))

    # Run clustering, inspect model-selection scores, and review cluster profiles.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Run the clustering workflow and display the selected cluster count and profile table.
                clustering_results = run_clustering(modeling_df, metadata)

                print(f"Best cluster count: {clustering_results['best_k']}")
                print(f"Best silhouette score: {clustering_results['best_score']:.4f}")

                display(clustering_results["silhouette_scores"])
                display(clustering_results["cluster_profile"])
                """
            )
        )
    )

    # Visualize the silhouette search and cluster composition for quick interpretation.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Visualize the silhouette scores and cluster sizes to interpret the segmentation result.
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                sns.barplot(
                    data=clustering_results["silhouette_scores"],
                    x="n_clusters",
                    y="silhouette_score",
                    ax=axes[0],
                    color="#4C78A8",
                )
                axes[0].set_title("Silhouette Score by Cluster Count")
                axes[0].set_xlabel("Number of Clusters")
                axes[0].set_ylabel("Silhouette Score")

                cluster_counts = clustering_results["clustered_df"]["cluster"].value_counts().sort_index()
                sns.barplot(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    ax=axes[1],
                    color="#F58518",
                )
                axes[1].set_title("Titles per Cluster")
                axes[1].set_xlabel("Cluster")
                axes[1].set_ylabel("Number of Titles")

                plt.tight_layout()
                plt.show()
                """
            )
        )
    )

    # Introduce the supervised content-type classification section.
    cells.append(nbf.v4.new_markdown_cell("## Content Type Classification"))

    # Run the type classifier and display the headline metrics and feature drivers.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Train the Movie-vs-TV-Show classifier and inspect the main evaluation outputs.
                classification_results = run_type_classification(modeling_df)

                print(f"Holdout accuracy: {classification_results['accuracy']:.4f}")
                print(f"Holdout weighted F1: {classification_results['weighted_f1']:.4f}")

                display(classification_results["confusion_matrix"])
                display(classification_results["top_features"])
                """
            )
        )
    )

    # Plot the strongest classification feature importances for easier discussion.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Plot the strongest permutation-based feature importances for the type classifier.
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=classification_results["top_features"].head(12),
                    x="importance",
                    y="feature",
                    color="#54A24B",
                )
                plt.title("Top Feature Drivers for Content Type Classification")
                plt.xlabel("Permutation Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.show()
                """
            )
        )
    )

    # Introduce the availability driver analysis design choice for multi-label fields.
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent_cell(
                """
                ## Driver Analysis for Countries and Genres

                Countries and genres are both multi-label fields in this dataset. To avoid forcing an arbitrary single label per title, the driver analysis below treats each top country and each top genre as a separate one-vs-rest prediction problem.
                """
            )
        )
    )

    # Run the shared driver-analysis helper and inspect the country-side results.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Run the availability driver analysis and inspect country-side performance and drivers.
                driver_results = run_availability_driver_analysis(modeling_df, metadata)

                print("Country driver model performance")
                display(driver_results["country"]["performance"])

                print("Average country driver importance")
                display(driver_results["country"]["average_importance"])
                """
            )
        )
    )

    # Plot country driver importances so the dominant cross-country signals stand out.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Visualize the strongest shared drivers behind title presence in the top countries.
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=driver_results["country"]["average_importance"].head(12),
                    x="importance",
                    y="feature",
                    color="#E45756",
                )
                plt.title("Average Feature Importance for Top-Country Availability")
                plt.xlabel("Average Permutation Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.show()
                """
            )
        )
    )

    # Inspect the genre-side driver tables before plotting them.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Inspect genre-side performance and the strongest average genre availability drivers.
                print("Genre driver model performance")
                display(driver_results["genre"]["performance"])

                print("Average genre driver importance")
                display(driver_results["genre"]["average_importance"])
                """
            )
        )
    )

    # Visualize genre availability drivers for an easier mentor-facing discussion.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Visualize the strongest shared drivers behind title presence in the top genres.
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=driver_results["genre"]["average_importance"].head(12),
                    x="importance",
                    y="feature",
                    color="#B279A2",
                )
                plt.title("Average Feature Importance for Top-Genre Availability")
                plt.xlabel("Average Permutation Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.show()
                """
            )
        )
    )

    # Save reusable output tables and print a compact final notebook summary.
    cells.append(
        nbf.v4.new_code_cell(
            dedent_cell(
                """
                # Save the shared Milestone 3 outputs and print a short interpretation summary.
                save_results(clustering_results, classification_results, driver_results)

                print("Milestone 3 notebook run complete.")
                print(
                    "Summary:",
                    {
                        "best_cluster_count": clustering_results["best_k"],
                        "best_silhouette_score": round(clustering_results["best_score"], 4),
                        "type_accuracy": round(classification_results["accuracy"], 4),
                        "type_weighted_f1": round(classification_results["weighted_f1"], 4),
                    },
                )
                """
            )
        )
    )

    # Attach the finished cell list and notebook metadata for execution.
    notebook.cells = cells
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {
        "name": "python",
        "version": "3.12",
    }
    return notebook


def main() -> None:
    """Write the generated notebook to disk."""
    # Build the notebook object and ensure the destination directory exists.
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Persist the notebook JSON so it can be executed with Jupyter tools.
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)

    # Print the destination path for quick command-line confirmation.
    print(f"Generated notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    # Allow direct script execution for notebook generation.
    main()
