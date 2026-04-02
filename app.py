from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/processed/netflix_titles_cleaned.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and enrich the cleaned Netflix dataset for dashboard use."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["genre_list"] = df["listed_in"].fillna("Unknown").apply(split_pipe_values)
    df["country_list"] = df["country"].fillna("Unknown").apply(split_pipe_values)
    return df


def split_pipe_values(value: str | None) -> list[str]:
    """Split a pipe-separated categorical field into cleaned labels."""
    if value is None or pd.isna(value):
        return ["Unknown"]

    values = sorted({item.strip() for item in str(value).split("|") if item.strip()})
    return values if values else ["Unknown"]


def matches_selected(values: list[str], selected: list[str]) -> bool:
    """Return True when a multi-value row matches the selected filter values."""
    if not selected:
        return True
    return any(value in selected for value in values)


def apply_filters(
    df: pd.DataFrame,
    *,
    years: list[int],
    genres: list[str],
    countries: list[str],
    content_types: list[str],
) -> pd.DataFrame:
    """Filter the dashboard dataset using the active sidebar selections."""
    filtered = df.copy()

    if years:
        filtered = filtered[filtered["release_year"].isin(years)]
    if content_types:
        filtered = filtered[filtered["type"].isin(content_types)]
    if genres:
        filtered = filtered[filtered["genre_list"].apply(lambda values: matches_selected(values, genres))]
    if countries:
        filtered = filtered[
            filtered["country_list"].apply(lambda values: matches_selected(values, countries))
        ]

    return filtered


def explode_counts(df: pd.DataFrame, source_column: str, output_name: str) -> pd.DataFrame:
    """Explode multi-value categorical lists and count title frequency."""
    if df.empty:
        return pd.DataFrame(columns=[output_name, "title_count"])

    exploded = df.explode(source_column)
    counts = (
        exploded.groupby(source_column)
        .size()
        .reset_index(name="title_count")
        .rename(columns={source_column: output_name})
        .sort_values("title_count", ascending=False)
    )
    return counts


def build_top_genres_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return the top genre for each release year in the filtered dataset."""
    if df.empty:
        return pd.DataFrame(columns=["release_year", "genre", "title_count"])

    genre_year = (
        df.explode("genre_list")
        .groupby(["release_year", "genre_list"])
        .size()
        .reset_index(name="title_count")
        .rename(columns={"genre_list": "genre"})
    )
    top_genres = (
        genre_year.sort_values(["release_year", "title_count", "genre"], ascending=[True, False, True])
        .groupby("release_year", as_index=False)
        .first()
        .sort_values("release_year")
    )
    return top_genres


def build_rating_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Return a matrix-friendly rating distribution split by content type."""
    if df.empty:
        return pd.DataFrame(columns=["rating", "type", "title_count"])

    return (
        df.groupby(["rating", "type"])
        .size()
        .reset_index(name="title_count")
        .sort_values(["title_count", "rating"], ascending=[False, True])
    )


def render_kpis(df: pd.DataFrame) -> None:
    """Render the headline dashboard metrics."""
    total_titles = int(df.shape[0])
    movie_count = int((df["type"] == "Movie").sum()) if not df.empty else 0
    tv_count = int((df["type"] == "TV Show").sum()) if not df.empty else 0
    median_release_year = int(df["release_year"].median()) if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Titles", f"{total_titles:,}")
    col2.metric("Movies", f"{movie_count:,}")
    col3.metric("TV Shows", f"{tv_count:,}")
    col4.metric("Median Release Year", median_release_year if median_release_year else "N/A")


def render_matplotlib_chart(fig: plt.Figure) -> None:
    """Render and close a Matplotlib figure in Streamlit."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(
        page_title="Netflix Dashboard",
        page_icon="N",
        layout="wide",
    )

    st.title("Netflix Content Dashboard")
    st.caption(
        "Interactive Milestone 4 dashboard built from the cleaned Netflix titles dataset."
    )

    data = load_data(str(DATA_PATH))

    available_years = sorted(data["release_year"].dropna().astype(int).unique().tolist())
    available_genres = sorted({genre for values in data["genre_list"] for genre in values})
    available_countries = sorted({country for values in data["country_list"] for country in values})
    available_types = sorted(data["type"].dropna().unique().tolist())

    st.sidebar.header("Filters")
    selected_years = st.sidebar.multiselect("Release Year", available_years)
    selected_genres = st.sidebar.multiselect("Genre", available_genres)
    selected_countries = st.sidebar.multiselect("Country", available_countries)
    selected_types = st.sidebar.multiselect("Content Type", available_types)

    filtered = apply_filters(
        data,
        years=selected_years,
        genres=selected_genres,
        countries=selected_countries,
        content_types=selected_types,
    )

    if filtered.empty:
        st.warning("No titles match the current filters. Adjust the sidebar selections to continue.")
        return

    render_kpis(filtered)

    st.subheader("Overview")
    overview_left, overview_right = st.columns(2)

    titles_by_year = (
        filtered.groupby("release_year")
        .size()
        .reset_index(name="title_count")
        .sort_values("release_year")
    )
    overview_left.caption("Titles by Release Year")
    fig_year, ax_year = plt.subplots(figsize=(8, 4))
    ax_year.plot(titles_by_year["release_year"], titles_by_year["title_count"], marker="o")
    ax_year.set_xlabel("Release Year")
    ax_year.set_ylabel("Titles")
    ax_year.set_title("Titles by Release Year")
    ax_year.grid(alpha=0.25)
    overview_left.pyplot(fig_year, use_container_width=True)
    plt.close(fig_year)

    type_counts = (
        filtered.groupby("type")
        .size()
        .reset_index(name="title_count")
        .sort_values("title_count", ascending=False)
    )
    overview_right.caption("Content Type Mix")
    fig_type, ax_type = plt.subplots(figsize=(6, 4))
    ax_type.pie(
        type_counts["title_count"],
        labels=type_counts["type"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_type.set_title("Content Type Mix")
    overview_right.pyplot(fig_type, use_container_width=True)
    plt.close(fig_type)

    st.subheader("Required Insights")
    insight_left, insight_right = st.columns(2)

    top_genres_by_year = build_top_genres_by_year(filtered)
    insight_left.caption("Top Genre Per Release Year")
    fig_genre, ax_genre = plt.subplots(figsize=(8, 4))
    ax_genre.bar(top_genres_by_year["release_year"].astype(str), top_genres_by_year["title_count"])
    ax_genre.set_xlabel("Release Year")
    ax_genre.set_ylabel("Titles")
    ax_genre.set_title("Top Genre Per Release Year")
    ax_genre.tick_params(axis="x", rotation=45)
    insight_left.pyplot(fig_genre, use_container_width=True)
    plt.close(fig_genre)
    insight_left.dataframe(
        top_genres_by_year,
        use_container_width=True,
        hide_index=True,
    )

    country_counts = explode_counts(filtered, "country_list", "country").head(15)
    insight_right.caption("Top Countries by Content Count")
    fig_country, ax_country = plt.subplots(figsize=(8, 5))
    reversed_country = country_counts.iloc[::-1]
    ax_country.barh(reversed_country["country"], reversed_country["title_count"])
    ax_country.set_xlabel("Titles")
    ax_country.set_ylabel("Country")
    ax_country.set_title("Top Countries by Content Count")
    insight_right.pyplot(fig_country, use_container_width=True)
    plt.close(fig_country)
    insight_right.dataframe(
        country_counts,
        use_container_width=True,
        hide_index=True,
    )

    rating_left, rating_right = st.columns(2)

    rating_counts = (
        filtered.groupby("rating")
        .size()
        .reset_index(name="title_count")
        .sort_values("title_count", ascending=False)
    )
    rating_left.caption("Rating Distribution")
    fig_rating, ax_rating = plt.subplots(figsize=(8, 4))
    ax_rating.bar(rating_counts["rating"], rating_counts["title_count"])
    ax_rating.set_xlabel("Rating")
    ax_rating.set_ylabel("Titles")
    ax_rating.set_title("Rating Distribution")
    ax_rating.tick_params(axis="x", rotation=45)
    rating_left.pyplot(fig_rating, use_container_width=True)
    plt.close(fig_rating)

    rating_by_type = build_rating_by_type(filtered)
    rating_right.caption("Rating Analysis by Content Type")
    rating_pivot = (
        rating_by_type.pivot(index="rating", columns="type", values="title_count")
        .fillna(0)
        .sort_index()
    )
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 5))
    image = ax_heatmap.imshow(rating_pivot.values, aspect="auto")
    ax_heatmap.set_xticks(range(len(rating_pivot.columns)))
    ax_heatmap.set_xticklabels(rating_pivot.columns)
    ax_heatmap.set_yticks(range(len(rating_pivot.index)))
    ax_heatmap.set_yticklabels(rating_pivot.index)
    ax_heatmap.set_title("Rating Analysis by Content Type")
    fig_heatmap.colorbar(image, ax=ax_heatmap, label="Titles")
    rating_right.pyplot(fig_heatmap, use_container_width=True)
    plt.close(fig_heatmap)

    st.subheader("Detailed Tables")
    table_left, table_right = st.columns(2)
    table_left.dataframe(top_genres_by_year, use_container_width=True, hide_index=True)
    table_right.dataframe(country_counts, use_container_width=True, hide_index=True)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered.drop(columns=["genre_list", "country_list"]).to_csv(index=False).encode("utf-8"),
        file_name="netflix_dashboard_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
