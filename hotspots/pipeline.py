from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point, box
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import PipelineConfig


LOGGER = logging.getLogger("hotspots.pipeline")
NEURAL_MODEL_IDS = {"rnn_sequence", "cnn_spatial"}
TREE_TABULAR_MODEL_IDS = {"bagging_tree", "random_forest", "extra_trees", "hist_gb", "xgboost"}
TRACK_IDS = ("all_crash", "severe")


@dataclass(frozen=True)
class TrackSpec:
    track_id: str
    target_column: str
    output_dir: Path
    benchmark_dir: Path
    kepler_dir: Path
    leaderboard_path: Path
    benchmark_metrics_path: Path
    benchmark_plot_path: Path
    best_model_metadata_path: Path
    best_model_predictions_path: Path
    model_path: Path
    explain_model_path: Path
    metrics_path: Path
    predictions_path: Path
    hotspots_path: Path
    metrics_plot_path: Path
    score_distribution_plot_path: Path
    hotspot_map_plot_path: Path
    temporal_heatmap_plot_path: Path
    shap_summary_path: Path
    hotspot_explanations_path: Path
    kepler_prediction_cells_path: Path
    kepler_hotspot_cells_path: Path
    kepler_config_path: Path
    kepler_html_path: Path
    leaflet_html_path: Path


def build_track_spec(config: PipelineConfig, track_id: str) -> TrackSpec:
    if track_id not in TRACK_IDS:
        raise ValueError(f"Unsupported track id: {track_id}")
    if track_id == "all_crash":
        output_dir = config.outputs_dir
        benchmark_dir = config.benchmark_dir
        kepler_dir = config.kepler_dir
        target_column = "label"
    else:
        output_dir = config.outputs_dir / "severe"
        benchmark_dir = output_dir / "benchmark"
        kepler_dir = output_dir / "kepler"
        target_column = "severe_label"
    return TrackSpec(
        track_id=track_id,
        target_column=target_column,
        output_dir=output_dir,
        benchmark_dir=benchmark_dir,
        kepler_dir=kepler_dir,
        leaderboard_path=benchmark_dir / "leaderboard.csv",
        benchmark_metrics_path=benchmark_dir / "metrics.json",
        benchmark_plot_path=benchmark_dir / "leaderboard.png",
        best_model_metadata_path=output_dir / "best_model.json",
        best_model_predictions_path=benchmark_dir / "best_model_test_predictions.csv",
        model_path=output_dir / "best_model.pkl",
        explain_model_path=output_dir / "explain_model.pkl",
        metrics_path=output_dir / "metrics.json",
        predictions_path=output_dir / "test_predictions.csv",
        hotspots_path=output_dir / "top_hotspots.csv",
        metrics_plot_path=output_dir / "metrics_summary.png",
        score_distribution_plot_path=output_dir / "prediction_score_distribution.png",
        hotspot_map_plot_path=output_dir / "top_hotspots_map.png",
        temporal_heatmap_plot_path=output_dir / "hourly_risk_heatmap.png",
        shap_summary_path=output_dir / "shap_summary.png",
        hotspot_explanations_path=output_dir / "hotspot_explanations.csv",
        kepler_prediction_cells_path=kepler_dir / "prediction_cells.geojson",
        kepler_hotspot_cells_path=kepler_dir / "top_hotspot_cells.geojson",
        kepler_config_path=kepler_dir / "kepler_config.json",
        kepler_html_path=kepler_dir / "nyc_hotspots_map.html",
        leaflet_html_path=kepler_dir / "nyc_hotspots_leaflet.html",
    )


def resolve_track_ids(targets: str | None = None) -> list[str]:
    if not targets:
        return list(TRACK_IDS)
    resolved = []
    for track_id in [part.strip() for part in targets.split(",") if part.strip()]:
        if track_id not in TRACK_IDS:
            raise ValueError(f"Unsupported target track: {track_id}")
        if track_id not in resolved:
            resolved.append(track_id)
    return resolved or list(TRACK_IDS)


def ensure_track_directories(track: TrackSpec) -> None:
    track.output_dir.mkdir(parents=True, exist_ok=True)
    track.benchmark_dir.mkdir(parents=True, exist_ok=True)
    track.kepler_dir.mkdir(parents=True, exist_ok=True)


def display_track_name(track_id: str) -> str:
    return "All Crash" if track_id == "all_crash" else "Severe"


def feature_display_name(feature_name: str) -> str:
    mapping = {
        "hour": "time of day",
        "day_of_week": "day of week",
        "month": "seasonality",
        "is_weekend": "weekend timing",
        "is_rush": "rush-hour timing",
        "is_night": "nighttime conditions",
        "centroid_lat": "north-south location",
        "centroid_lng": "east-west location",
        "prior_cell_crash_count": "prior crash history",
        "rolling_recent_crash_count": "recent crash recurrence",
        "prior_cell_severe_count": "prior severe crash history",
        "prior_cell_hour_crash_count": "same-hour recurrence",
        "prior_cell_dow_hour_crash_count": "same day/hour recurrence",
        "hours_since_last_cell_crash": "recency since last crash",
        "prior_avg_severity": "historical crash severity",
        "prior_night_rate": "historical night crash share",
        "historic_cell_crash_total": "long-run cell crash volume",
        "historic_neighbor_crash_total": "neighboring crash volume",
        "historic_hotspot_density": "neighboring hotspot density",
        "observed_neighbor_count": "nearby road-cell density",
        "nypd_prior_collision_count": "prior NYPD collision history",
        "nypd_prior_injury_count": "prior NYPD injury history",
        "nypd_prior_killed_count": "prior NYPD fatality history",
        "nypd_prior_aggressive_count": "aggressive driving history",
        "nypd_prior_distracted_count": "distracted driving history",
        "nypd_prior_yield_count": "failure-to-yield history",
        "nypd_prior_speed_count": "speeding history",
        "nypd_prior_alcohol_count": "alcohol/drug history",
        "weather_rain_history": "rain-related crash history",
        "weather_snow_history": "snow-related crash history",
        "weather_fog_history": "fog-related crash history",
        "borough_bronx": "Bronx location",
        "borough_brooklyn": "Brooklyn location",
        "borough_manhattan": "Manhattan location",
        "borough_queens": "Queens location",
        "borough_staten_island": "Staten Island location",
    }
    return mapping.get(feature_name, feature_name.replace("_", " "))


def compose_cause_summary(feature_names: list[str]) -> str:
    if not feature_names:
        return "High predicted risk is driven by a diffuse mix of model signals."
    readable = [feature_display_name(name) for name in feature_names]
    if len(readable) == 1:
        joined = readable[0]
    elif len(readable) == 2:
        joined = " and ".join(readable)
    else:
        joined = ", ".join(readable[:-1]) + f", and {readable[-1]}"
    return f"High predicted risk is mainly driven by {joined}."


def normalize_shap_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 3 and matrix.shape[-1] == 2:
        matrix = matrix[:, :, 1]
    if matrix.ndim != 2:
        raise ValueError(f"Unsupported SHAP value shape: {matrix.shape}")
    return matrix


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_directories(config: PipelineConfig) -> None:
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    config.benchmark_dir.mkdir(parents=True, exist_ok=True)
    config.kepler_dir.mkdir(parents=True, exist_ok=True)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_weather_category(series: pd.Series) -> pd.Series:
    text = normalize_text(series)
    category = pd.Series("other", index=series.index, dtype="object")
    category = category.mask(text.str.contains("rain|drizzle|storm|shower|thunder", na=False), "rain")
    category = category.mask(text.str.contains("snow|sleet|hail|wintry", na=False), "snow")
    category = category.mask(text.str.contains("fog|mist|haze|smoke", na=False), "fog")
    return category


def borough_from_parts(city_norm: pd.Series, county_norm: pd.Series) -> pd.Series:
    borough = pd.Series("unknown", index=city_norm.index, dtype="object")
    county_map = {
        "bronx": "bronx",
        "kings": "brooklyn",
        "new york": "manhattan",
        "queens": "queens",
        "richmond": "staten_island",
    }
    city_map = {
        "bronx": "bronx",
        "brooklyn": "brooklyn",
        "manhattan": "manhattan",
        "new york": "manhattan",
        "queens": "queens",
        "staten island": "staten_island",
    }
    for key, value in county_map.items():
        borough = borough.mask(county_norm == key, value)
    for key, value in city_map.items():
        borough = borough.mask(city_norm == key, value)
    return borough


def normalize_factor_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def classify_factor_text(text: pd.Series, category: str) -> pd.Series:
    if category == "aggressive":
        pattern = r"aggressive|road rage|tailgating|passing|following too closely"
    elif category == "distracted":
        pattern = r"distracted|inattention|cell phone|electronic|using on board navigation"
    elif category == "yield":
        pattern = r"yield|failure to yield|right-of-way"
    elif category == "speed":
        pattern = r"speed|unsafe speed|too fast"
    elif category == "alcohol":
        pattern = r"alcohol|drugs|intox"
    else:
        pattern = category
    return text.str.contains(pattern, regex=True, na=False).astype(int)


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["year"] = result["bucket_start"].dt.year.astype(int)
    result["month"] = result["bucket_start"].dt.month.astype(int)
    result["day_of_week"] = result["bucket_start"].dt.dayofweek.astype(int)
    result["hour"] = result["bucket_start"].dt.hour.astype(int)
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    result["is_rush"] = result["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    return result


def compute_top_capture_rate(y_true: pd.Series, scores: pd.Series, fraction: float) -> float | None:
    if y_true.empty or y_true.sum() == 0:
        return None
    top_n = max(1, math.ceil(len(scores) * fraction))
    ranked = pd.DataFrame({"label": y_true, "score": scores}).sort_values("score", ascending=False)
    captured = ranked.head(top_n)["label"].sum()
    return float(captured / y_true.sum())


def build_prediction_frame(
    df: pd.DataFrame,
    scores: np.ndarray,
    target_column: str,
    track_id: str,
) -> pd.DataFrame:
    frame = df[
        [
            "row_id",
            "cell_id",
            "bucket_start",
            "year",
            "month",
            "day_of_week",
            "hour",
            "centroid_lat",
            "centroid_lng",
            "split",
        ]
    ].copy()
    if "label" in df.columns:
        frame["all_crash_label"] = df["label"].astype(int)
    if "severe_label" in df.columns:
        frame["severe_label"] = df["severe_label"].astype(int)
    frame["target_track"] = track_id
    frame["target_column"] = target_column
    frame["label"] = df[target_column].astype(int)
    frame["predicted_probability"] = scores
    frame["predicted_label"] = (frame["predicted_probability"] >= 0.5).astype(int)
    return frame.sort_values("predicted_probability", ascending=False).reset_index(drop=True)


def aggregate_hotspots(predictions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    hotspots = (
        predictions.groupby(["cell_id", "centroid_lat", "centroid_lng"], as_index=False)
        .agg(
            mean_probability=("predicted_probability", "mean"),
            max_probability=("predicted_probability", "max"),
            bucket_count=("cell_id", "size"),
            observed_positive_buckets=("label", "sum"),
        )
        .sort_values(["mean_probability", "max_probability"], ascending=False)
        .reset_index(drop=True)
    )
    top_n = max(1, math.ceil(len(hotspots) * config.top_hotspot_fraction))
    return hotspots.head(top_n).copy()


def evaluate_predictions(labels: pd.Series, scores: np.ndarray, config: PipelineConfig) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "roc_auc": None,
        "average_precision": None,
        "top_5pct_capture": compute_top_capture_rate(labels, pd.Series(scores), config.top_hotspot_fraction),
    }
    if labels.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    return metrics


def prepare_data(config: PipelineConfig) -> Path:
    ensure_directories(config)
    if not config.raw_accidents_path.exists():
        raise FileNotFoundError(f"Missing input data file: {config.raw_accidents_path}")

    filtered_chunks: list[pd.DataFrame] = []
    total_rows = 0
    ny_rows = 0
    nyc_rows = 0

    for chunk in pd.read_csv(
        config.raw_accidents_path,
        usecols=list(config.required_columns),
        chunksize=config.chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk = chunk[chunk["State"] == "NY"].copy()
        ny_rows += len(chunk)
        if chunk.empty:
            continue

        chunk["city_norm"] = normalize_text(chunk["City"])
        chunk["county_norm"] = normalize_text(chunk["County"])
        is_nyc = chunk["county_norm"].isin(config.nyc_counties) | chunk["city_norm"].isin(config.nyc_cities)
        chunk = chunk.loc[is_nyc].copy()
        nyc_rows += len(chunk)
        if chunk.empty:
            continue

        chunk["Start_Time"] = pd.to_datetime(chunk["Start_Time"], errors="coerce")
        chunk = chunk.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng"])
        chunk["year"] = chunk["Start_Time"].dt.year
        chunk = chunk[chunk["year"].between(config.min_year, config.max_year)].copy()
        if chunk.empty:
            continue

        chunk["borough"] = borough_from_parts(chunk["city_norm"], chunk["county_norm"])
        chunk["weather_category"] = normalize_weather_category(chunk["Weather_Condition"])
        filtered_chunks.append(
            chunk[
                [
                    "City",
                    "County",
                    "Start_Time",
                    "Start_Lat",
                    "Start_Lng",
                    "Severity",
                    "Sunrise_Sunset",
                    "Weather_Condition",
                    "weather_category",
                    "borough",
                    "year",
                ]
            ]
        )

    if not filtered_chunks:
        raise ValueError("No NYC accident rows were retained from the raw accident file.")

    prepared = pd.concat(filtered_chunks, ignore_index=True)
    prepared["Severity"] = pd.to_numeric(prepared["Severity"], errors="coerce").fillna(0).astype(int)
    prepared["Sunrise_Sunset"] = normalize_text(prepared["Sunrise_Sunset"])

    LOGGER.info(
        "Prepared accidents: total_rows=%s ny_rows=%s nyc_rows=%s retained=%s years=%s",
        total_rows,
        ny_rows,
        nyc_rows,
        len(prepared),
        prepared["year"].value_counts().sort_index().to_dict(),
    )
    prepared.to_parquet(config.prepared_path, index=False)
    return config.prepared_path


def build_grid(config: PipelineConfig) -> Path:
    ensure_directories(config)
    df = pd.read_parquet(config.prepared_path)
    if df.empty:
        raise ValueError("Prepared accidents dataset is empty.")

    geometry = gpd.points_from_xy(df["Start_Lng"], df["Start_Lat"])
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=config.output_crs).to_crs(config.processing_crs)
    gdf["grid_x"] = np.floor(gdf.geometry.x / config.grid_size_meters).astype("int64")
    gdf["grid_y"] = np.floor(gdf.geometry.y / config.grid_size_meters).astype("int64")
    gdf["cell_id"] = gdf["grid_x"].astype(str) + "_" + gdf["grid_y"].astype(str)
    gdf["bucket_start"] = gdf["Start_Time"].dt.floor("h")
    gdf["is_night_event"] = (gdf["Sunrise_Sunset"] == "night").astype(int)
    gdf["is_severe_event"] = (gdf["Severity"] >= 3).astype(int)
    gdf["weather_rain_event"] = (gdf["weather_category"] == "rain").astype(int)
    gdf["weather_snow_event"] = (gdf["weather_category"] == "snow").astype(int)
    gdf["weather_fog_event"] = (gdf["weather_category"] == "fog").astype(int)

    grouped = (
        gdf.groupby(["cell_id", "grid_x", "grid_y", "borough", "bucket_start"], as_index=False)
        .agg(
            crash_count=("Severity", "size"),
            severe_count=("is_severe_event", "sum"),
            severity_sum=("Severity", "sum"),
            max_severity=("Severity", "max"),
            night_event_count=("is_night_event", "sum"),
            weather_rain_count=("weather_rain_event", "sum"),
            weather_snow_count=("weather_snow_event", "sum"),
            weather_fog_count=("weather_fog_event", "sum"),
        )
        .sort_values(["bucket_start", "cell_id"])
        .reset_index(drop=True)
    )
    grouped["label"] = 1
    grouped["severe_label"] = (grouped["severe_count"] > 0).astype(int)
    grouped["centroid_x"] = (grouped["grid_x"] + 0.5) * config.grid_size_meters
    grouped["centroid_y"] = (grouped["grid_y"] + 0.5) * config.grid_size_meters
    centroid_geom = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(grouped["centroid_x"], grouped["centroid_y"])],
        crs=config.processing_crs,
    ).to_crs(config.output_crs)
    grouped["centroid_lng"] = centroid_geom.x
    grouped["centroid_lat"] = centroid_geom.y
    grouped["avg_severity"] = grouped["severity_sum"] / grouped["crash_count"].clip(lower=1)
    grouped = grouped.drop(columns=["centroid_x", "centroid_y"])
    grouped = add_time_columns(grouped)

    cell_catalog = (
        grouped[["cell_id", "grid_x", "grid_y", "borough", "centroid_lat", "centroid_lng"]]
        .sort_values(["cell_id", "borough"])
        .drop_duplicates(subset=["cell_id"], keep="first")
        .reset_index(drop=True)
    )
    cell_catalog.to_parquet(config.cell_catalog_path, index=False)

    LOGGER.info(
        "Grid built with %s hourly positive rows across %s unique cells",
        len(grouped),
        grouped["cell_id"].nunique(),
    )
    grouped.to_parquet(config.grid_path, index=False)
    return config.grid_path


def sample_negative_rows(
    positive: pd.DataFrame,
    cell_catalog: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    positive_keys = positive[["cell_id", "bucket_start"]].drop_duplicates()
    min_time = positive["bucket_start"].min()
    max_time = positive["bucket_start"].max()
    timeline = pd.date_range(min_time, max_time, freq="h")
    target_count = len(positive) * config.negative_pool_ratio
    max_unique = len(cell_catalog) * len(timeline) - len(positive_keys)
    target_count = min(target_count, max(0, max_unique))
    if target_count == 0:
        raise ValueError("Unable to generate negative rows because the candidate space is exhausted.")

    batches: list[pd.DataFrame] = []
    collected = 0
    attempts = 0
    batch_size = max(10_000, min(200_000, target_count * 2))

    while collected < target_count and attempts < 30:
        attempts += 1
        cell_idx = rng.integers(0, len(cell_catalog), size=batch_size)
        time_idx = rng.integers(0, len(timeline), size=batch_size)
        sampled_cells = cell_catalog.iloc[cell_idx].reset_index(drop=True)
        sampled_times = pd.DataFrame({"bucket_start": timeline[time_idx]})
        candidates = pd.concat([sampled_cells, sampled_times], axis=1).drop_duplicates()
        candidates = candidates.merge(
            positive_keys.assign(_positive=1),
            on=["cell_id", "bucket_start"],
            how="left",
        )
        candidates = candidates[candidates["_positive"].isna()].drop(columns="_positive")
        if candidates.empty:
            continue
        batches.append(candidates)
        collected += len(candidates)

    negatives = pd.concat(batches, ignore_index=True).drop_duplicates()
    if len(negatives) < target_count:
        LOGGER.warning("Negative sampling requested %s rows but generated %s", target_count, len(negatives))
        target_count = len(negatives)
    negatives = negatives.sample(n=target_count, random_state=config.random_seed).reset_index(drop=True)
    negatives["label"] = 0
    negatives["severe_label"] = 0
    negatives["crash_count"] = 0
    negatives["severe_count"] = 0
    negatives["severity_sum"] = 0
    negatives["max_severity"] = 0
    negatives["avg_severity"] = 0.0
    negatives["night_event_count"] = 0
    negatives["weather_rain_count"] = 0
    negatives["weather_snow_count"] = 0
    negatives["weather_fog_count"] = 0
    negatives = add_time_columns(negatives)
    return negatives


def load_nypd_hourly(config: PipelineConfig) -> pd.DataFrame:
    ensure_directories(config)
    if config.nypd_hourly_path.exists():
        return pd.read_parquet(config.nypd_hourly_path)
    if not config.raw_nypd_path.exists():
        columns = ["cell_id", "bucket_start"] + [f"nypd_{name}" for name in ["collision_count", "injury_count", "killed_count"]]
        return pd.DataFrame(columns=columns)

    usecols = [
        "CRASH DATE",
        "CRASH TIME",
        "LATITUDE",
        "LONGITUDE",
        "NUMBER OF PERSONS INJURED",
        "NUMBER OF PERSONS KILLED",
        "CONTRIBUTING FACTOR VEHICLE 1",
        "CONTRIBUTING FACTOR VEHICLE 2",
        "CONTRIBUTING FACTOR VEHICLE 3",
        "CONTRIBUTING FACTOR VEHICLE 4",
        "CONTRIBUTING FACTOR VEHICLE 5",
        "COLLISION_ID",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        config.raw_nypd_path,
        usecols=usecols,
        chunksize=config.nypd_chunksize,
        low_memory=False,
    ):
        chunk["LATITUDE"] = pd.to_numeric(chunk["LATITUDE"], errors="coerce")
        chunk["LONGITUDE"] = pd.to_numeric(chunk["LONGITUDE"], errors="coerce")
        chunk = chunk.dropna(subset=["LATITUDE", "LONGITUDE", "CRASH DATE", "CRASH TIME"])
        crash_dt = pd.to_datetime(
            chunk["CRASH DATE"].astype(str) + " " + chunk["CRASH TIME"].astype(str),
            errors="coerce",
        )
        chunk["bucket_start"] = crash_dt.dt.floor("h")
        chunk = chunk.dropna(subset=["bucket_start"])
        chunk["year"] = chunk["bucket_start"].dt.year
        chunk = chunk[chunk["year"].between(config.min_year, config.max_year)].copy()
        if chunk.empty:
            continue

        chunk["NUMBER OF PERSONS INJURED"] = pd.to_numeric(
            chunk["NUMBER OF PERSONS INJURED"],
            errors="coerce",
        ).fillna(0)
        chunk["NUMBER OF PERSONS KILLED"] = pd.to_numeric(
            chunk["NUMBER OF PERSONS KILLED"],
            errors="coerce",
        ).fillna(0)
        factor_cols = [
            "CONTRIBUTING FACTOR VEHICLE 1",
            "CONTRIBUTING FACTOR VEHICLE 2",
            "CONTRIBUTING FACTOR VEHICLE 3",
            "CONTRIBUTING FACTOR VEHICLE 4",
            "CONTRIBUTING FACTOR VEHICLE 5",
        ]
        factor_text = normalize_text(chunk[factor_cols].fillna("").agg(" | ".join, axis=1))
        for category in config.nypd_factor_categories:
            chunk[f"factor_{category}"] = classify_factor_text(factor_text, category)

        geometry = gpd.points_from_xy(chunk["LONGITUDE"], chunk["LATITUDE"])
        gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs=config.output_crs).to_crs(config.processing_crs)
        gdf["grid_x"] = np.floor(gdf.geometry.x / config.grid_size_meters).astype("int64")
        gdf["grid_y"] = np.floor(gdf.geometry.y / config.grid_size_meters).astype("int64")
        gdf["cell_id"] = gdf["grid_x"].astype(str) + "_" + gdf["grid_y"].astype(str)
        aggregated = (
            gdf.groupby(["cell_id", "bucket_start"], as_index=False)
            .agg(
                nypd_collision_count=("COLLISION_ID", "size"),
                nypd_injury_count=("NUMBER OF PERSONS INJURED", "sum"),
                nypd_killed_count=("NUMBER OF PERSONS KILLED", "sum"),
                nypd_aggressive_count=("factor_aggressive", "sum"),
                nypd_distracted_count=("factor_distracted", "sum"),
                nypd_yield_count=("factor_yield", "sum"),
                nypd_speed_count=("factor_speed", "sum"),
                nypd_alcohol_count=("factor_alcohol", "sum"),
            )
        )
        chunks.append(aggregated)

    if chunks:
        nypd_hourly = (
            pd.concat(chunks, ignore_index=True)
            .groupby(["cell_id", "bucket_start"], as_index=False)
            .sum(numeric_only=True)
        )
    else:
        nypd_hourly = pd.DataFrame(
            columns=[
                "cell_id",
                "bucket_start",
                "nypd_collision_count",
                "nypd_injury_count",
                "nypd_killed_count",
                "nypd_aggressive_count",
                "nypd_distracted_count",
                "nypd_yield_count",
                "nypd_speed_count",
                "nypd_alcohol_count",
            ]
        )
    nypd_hourly.to_parquet(config.nypd_hourly_path, index=False)
    return nypd_hourly


def compute_static_cell_features(grid: pd.DataFrame, cell_catalog: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    historic = grid[grid["year"] < config.test_year].copy()
    historic_cell = (
        historic.groupby("cell_id", as_index=False)
        .agg(historic_cell_crash_total=("crash_count", "sum"))
    )
    deduped_catalog = (
        cell_catalog.sort_values(["cell_id", "borough"])
        .drop_duplicates(subset=["cell_id"], keep="first")
        .reset_index(drop=True)
    )
    duplicate_count = len(cell_catalog) - len(deduped_catalog)
    if duplicate_count > 0:
        LOGGER.warning("Deduplicated %s repeated cell_id rows from cell catalog", duplicate_count)
    feature_frame = deduped_catalog.merge(historic_cell, on="cell_id", how="left").fillna(
        {"historic_cell_crash_total": 0}
    )
    crash_lookup = feature_frame.set_index("cell_id")["historic_cell_crash_total"].to_dict()
    coord_lookup = feature_frame.set_index("cell_id")[["grid_x", "grid_y"]].to_dict("index")
    reverse_lookup = {(row["grid_x"], row["grid_y"]): row["cell_id"] for _, row in feature_frame.iterrows()}
    neighbor_records: list[dict[str, Any]] = []
    for cell_id, coords in coord_lookup.items():
        neighbor_ids: list[str] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = reverse_lookup.get((coords["grid_x"] + dx, coords["grid_y"] + dy))
                if neighbor is not None:
                    neighbor_ids.append(neighbor)
        neighbor_total = float(sum(crash_lookup.get(neighbor, 0.0) for neighbor in neighbor_ids))
        neighbor_density = float(
            sum(1 for neighbor in neighbor_ids if crash_lookup.get(neighbor, 0.0) > 0) / max(1, len(neighbor_ids))
        )
        neighbor_records.append(
            {
                "cell_id": cell_id,
                "observed_neighbor_count": len(neighbor_ids),
                "historic_neighbor_crash_total": neighbor_total,
                "historic_hotspot_density": neighbor_density,
            }
        )
    neighbors = pd.DataFrame(neighbor_records)
    return feature_frame.merge(neighbors, on="cell_id", how="left")


def compute_hours_since_last_crash(df: pd.DataFrame) -> pd.Series:
    last_positive_time = df["bucket_start"].where(df["crash_count"] > 0)
    last_positive_time = last_positive_time.groupby(df["cell_id"]).ffill().groupby(df["cell_id"]).shift()
    hours = (df["bucket_start"] - last_positive_time).dt.total_seconds() / 3600.0
    return hours.fillna(-1.0)


def engineer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values(["cell_id", "bucket_start", "label"]).reset_index(drop=True).copy()
    result["is_night"] = (result["night_event_count"] > 0).astype(int)
    result["prior_cell_crash_count"] = result.groupby("cell_id")["crash_count"].cumsum() - result["crash_count"]
    result["rolling_recent_crash_count"] = (
        result.groupby("cell_id")["crash_count"]
        .transform(lambda series: series.shift().rolling(window=3, min_periods=1).sum())
        .fillna(0)
    )
    result["prior_cell_severe_count"] = result.groupby("cell_id")["severe_count"].cumsum() - result["severe_count"]
    result["prior_cell_hour_crash_count"] = (
        result.groupby(["cell_id", "hour"])["crash_count"].cumsum() - result["crash_count"]
    )
    result["prior_cell_dow_hour_crash_count"] = (
        result.groupby(["cell_id", "day_of_week", "hour"])["crash_count"].cumsum() - result["crash_count"]
    )
    result["prior_severity_sum"] = result.groupby("cell_id")["severity_sum"].cumsum() - result["severity_sum"]
    prior_night_total = result.groupby("cell_id")["night_event_count"].cumsum() - result["night_event_count"]
    result["prior_avg_severity"] = np.where(
        result["prior_cell_crash_count"] > 0,
        result["prior_severity_sum"] / result["prior_cell_crash_count"],
        0.0,
    )
    result["prior_night_rate"] = np.where(
        result["prior_cell_crash_count"] > 0,
        prior_night_total / result["prior_cell_crash_count"],
        0.0,
    )
    result["weather_rain_history"] = result.groupby("cell_id")["weather_rain_count"].cumsum() - result["weather_rain_count"]
    result["weather_snow_history"] = result.groupby("cell_id")["weather_snow_count"].cumsum() - result["weather_snow_count"]
    result["weather_fog_history"] = result.groupby("cell_id")["weather_fog_count"].cumsum() - result["weather_fog_count"]
    for base_name in [
        "nypd_collision_count",
        "nypd_injury_count",
        "nypd_killed_count",
        "nypd_aggressive_count",
        "nypd_distracted_count",
        "nypd_yield_count",
        "nypd_speed_count",
        "nypd_alcohol_count",
    ]:
        result[f"{base_name.replace('nypd_', 'nypd_prior_')}"] = (
            result.groupby("cell_id")[base_name].cumsum() - result[base_name]
        )
    result["hours_since_last_cell_crash"] = compute_hours_since_last_crash(result)
    result["delta_hours"] = (
        result.groupby("cell_id")["bucket_start"].diff().dt.total_seconds().div(3600.0).fillna(0.0)
    )
    borough_dummies = pd.get_dummies(result["borough"], prefix="borough")
    result = pd.concat([result, borough_dummies], axis=1)
    for borough_column in [
        "borough_bronx",
        "borough_brooklyn",
        "borough_manhattan",
        "borough_queens",
        "borough_staten_island",
    ]:
        if borough_column not in result.columns:
            result[borough_column] = 0
    return result


def build_tabular_features(config: PipelineConfig) -> Path:
    ensure_directories(config)
    positive = pd.read_parquet(config.grid_path)
    if positive.empty:
        raise ValueError("Grid accidents dataset is empty.")
    cell_catalog = pd.read_parquet(config.cell_catalog_path)
    static_features = compute_static_cell_features(positive, cell_catalog, config)
    negatives = sample_negative_rows(positive, cell_catalog, config)
    combined = pd.concat([positive, negatives], ignore_index=True, sort=False)
    combined = add_time_columns(combined)
    combined = combined.merge(
        static_features[
            [
                "cell_id",
                "historic_cell_crash_total",
                "historic_neighbor_crash_total",
                "historic_hotspot_density",
                "observed_neighbor_count",
            ]
        ],
        on="cell_id",
        how="left",
    )
    nypd_hourly = load_nypd_hourly(config)
    if not nypd_hourly.empty:
        combined = combined.merge(nypd_hourly, on=["cell_id", "bucket_start"], how="left")
    for column in [
        "nypd_collision_count",
        "nypd_injury_count",
        "nypd_killed_count",
        "nypd_aggressive_count",
        "nypd_distracted_count",
        "nypd_yield_count",
        "nypd_speed_count",
        "nypd_alcohol_count",
    ]:
        if column not in combined.columns:
            combined[column] = 0
        combined[column] = combined[column].fillna(0)

    combined["historic_cell_crash_total"] = combined["historic_cell_crash_total"].fillna(0)
    combined["historic_neighbor_crash_total"] = combined["historic_neighbor_crash_total"].fillna(0)
    combined["historic_hotspot_density"] = combined["historic_hotspot_density"].fillna(0.0)
    combined["observed_neighbor_count"] = combined["observed_neighbor_count"].fillna(0)
    features = engineer_history_features(combined)
    features = features.fillna(0)

    missing = [column for column in config.tabular_feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Missing engineered feature columns: {missing}")
    config.feature_columns_path.write_text(json.dumps(list(config.tabular_feature_columns), indent=2) + "\n")
    LOGGER.info(
        "Tabular features built with %s rows and %s engineered columns",
        len(features),
        len(config.tabular_feature_columns),
    )
    features.to_parquet(config.features_path, index=False)
    return config.features_path


def stratified_historic_sample(df: pd.DataFrame, fraction: float, random_seed: int) -> pd.DataFrame:
    sampled_groups: list[pd.DataFrame] = []
    for _, group in df.groupby(["month", "day_of_week"], dropna=False):
        sample_size = max(1, int(math.ceil(len(group) * fraction)))
        sample_size = min(sample_size, len(group))
        sampled_groups.append(group.sample(n=sample_size, random_state=random_seed))
    return pd.concat(sampled_groups, ignore_index=True) if sampled_groups else df.iloc[0:0].copy()


def balance_training_rows(df: pd.DataFrame, ratio: int, random_seed: int) -> pd.DataFrame:
    positive = df[df["label"] == 1]
    negative = df[df["label"] == 0]
    if positive.empty or negative.empty:
        raise ValueError("Training split must contain both positive and negative rows.")
    target_negatives = min(len(negative), len(positive) * ratio)
    negative = negative.sample(n=target_negatives, random_state=random_seed)
    balanced = pd.concat([positive, negative], ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def sample_training_data(config: PipelineConfig) -> Path:
    ensure_directories(config)
    features = pd.read_parquet(config.features_path)
    if features.empty:
        raise ValueError("Features dataset is empty.")
    recent = features[features["year"] == config.test_year].copy()
    historic = features[features["year"] < config.test_year].copy()
    if recent.empty or historic.empty:
        raise ValueError("Feature dataset must contain both historic rows and test-year rows.")

    historic_sample = stratified_historic_sample(
        historic,
        fraction=config.historical_sample_fraction,
        random_seed=config.random_seed,
    )
    balanced_historic = balance_training_rows(
        historic_sample,
        ratio=config.positive_negative_ratio,
        random_seed=config.random_seed,
    )
    train_df, validation_df = train_test_split(
        balanced_historic,
        test_size=config.validation_fraction,
        random_state=config.random_seed,
        stratify=balanced_historic["label"],
    )
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    recent = recent.copy()
    train_df["split"] = "train"
    validation_df["split"] = "validation"
    recent["split"] = "test"
    sample = pd.concat([train_df, validation_df, recent], ignore_index=True)
    sample = sample.sort_values(["bucket_start", "cell_id", "label"]).reset_index(drop=True)
    sample["row_id"] = np.arange(len(sample), dtype=int)
    LOGGER.info(
        "Sample built with train=%s validation=%s test=%s",
        len(train_df),
        len(validation_df),
        len(recent),
    )
    sample.to_parquet(config.sample_path, index=False)
    return config.sample_path


def build_sequence_data(config: PipelineConfig) -> Path:
    ensure_directories(config)
    sample = pd.read_parquet(config.sample_path).sort_values(["cell_id", "bucket_start", "row_id"]).reset_index(drop=True)
    feature_cols = list(config.sequence_feature_columns)
    lookback = config.neural_lookback
    feature_count = len(feature_cols)
    sequence_data = np.zeros((len(sample), lookback, feature_count), dtype=np.float32)

    for _, group in sample.groupby("cell_id", sort=False):
        group_features = group[feature_cols].to_numpy(dtype=np.float32)
        group_indices = group.index.to_numpy()
        for position, global_idx in enumerate(group_indices):
            start = max(0, position - lookback + 1)
            window = group_features[start : position + 1]
            sequence_data[global_idx, -len(window) :, :] = window

    payload: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        mask = sample["split"] == split
        payload[f"{split}_X"] = sequence_data[mask.to_numpy()]
        payload[f"{split}_y"] = sample.loc[mask, "label"].to_numpy(dtype=np.float32)
        payload[f"{split}_row_id"] = sample.loc[mask, "row_id"].to_numpy(dtype=np.int64)
    np.savez_compressed(config.sequence_dataset_path, **payload)
    config.sequence_metadata_path.write_text(
        json.dumps(
            {
                "lookback": lookback,
                "feature_columns": feature_cols,
                "shape": {key: list(value.shape) for key, value in payload.items() if key.endswith("_X")},
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Sequence data saved to %s", config.sequence_dataset_path)
    return config.sequence_dataset_path


def build_spatial_data(config: PipelineConfig) -> Path:
    ensure_directories(config)
    sample = pd.read_parquet(config.sample_path).sort_values(["bucket_start", "cell_id", "row_id"]).reset_index(drop=True)
    features = pd.read_parquet(config.features_path)
    channels = list(config.spatial_patch_channels)
    patch_radius = config.spatial_patch_radius
    patch_size = patch_radius * 2 + 1

    feature_lookup: dict[tuple[str, int], np.ndarray] = {}
    for row in features[["cell_id", "bucket_start", *channels]].itertuples(index=False):
        cell_id = row[0]
        timestamp_key = int(pd.Timestamp(row[1]).value)
        feature_lookup[(cell_id, timestamp_key)] = np.asarray(row[2:], dtype=np.float32)

    cell_catalog = pd.read_parquet(config.cell_catalog_path)
    reverse_lookup = {
        (int(row.grid_x), int(row.grid_y)): row.cell_id for row in cell_catalog.itertuples(index=False)
    }
    patch_data = np.zeros((len(sample), len(channels), patch_size, patch_size), dtype=np.float32)
    for row in sample.itertuples(index=False):
        row_idx = int(row.row_id)
        timestamp_key = int(pd.Timestamp(row.bucket_start).value)
        for dx in range(-patch_radius, patch_radius + 1):
            for dy in range(-patch_radius, patch_radius + 1):
                neighbor_cell = reverse_lookup.get((int(row.grid_x) + dx, int(row.grid_y) + dy))
                if neighbor_cell is None:
                    continue
                values = feature_lookup.get((neighbor_cell, timestamp_key))
                if values is None:
                    continue
                patch_data[row_idx, :, dx + patch_radius, dy + patch_radius] = values

    payload: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        mask = sample["split"] == split
        payload[f"{split}_X"] = patch_data[mask.to_numpy()]
        payload[f"{split}_y"] = sample.loc[mask, "label"].to_numpy(dtype=np.float32)
        payload[f"{split}_row_id"] = sample.loc[mask, "row_id"].to_numpy(dtype=np.int64)
    np.savez_compressed(config.spatial_dataset_path, **payload)
    config.spatial_metadata_path.write_text(
        json.dumps(
            {
                "patch_radius": patch_radius,
                "channels": channels,
                "shape": {key: list(value.shape) for key, value in payload.items() if key.endswith("_X")},
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Spatial patch data saved to %s", config.spatial_dataset_path)
    return config.spatial_dataset_path


def build_tabular_model(model_id: str, y_train: pd.Series, config: PipelineConfig) -> Any:
    if model_id == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.random_seed)),
            ]
        )
    if model_id == "svm_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=config.random_seed,
                        cache_size=500,
                    ),
                ),
            ]
        )
    if model_id == "bagging_tree":
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=config.random_seed),
            n_estimators=40,
            random_state=config.random_seed,
            n_jobs=-1,
        )
    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=config.random_seed,
            n_jobs=-1,
        )
    if model_id == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
        )
    if model_id == "hist_gb":
        return HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=250,
            learning_rate=0.05,
            random_state=config.random_seed,
        )
    if model_id == "xgboost":
        import xgboost as xgb

        positives = max(1, int((y_train == 1).sum()))
        negatives = max(1, int((y_train == 0).sum()))
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=negatives / positives,
            eval_metric="auc",
            random_state=config.random_seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported tabular model id: {model_id}")


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Model does not support probability-like scoring.")


def maybe_import_torch() -> Any:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        return torch, nn, DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for neural benchmark models. Install `torch` first.") from exc


def prepare_neural_arrays(train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (val_X - mean) / std, (test_X - mean) / std


def train_rnn_model(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch, nn, DataLoader, TensorDataset = maybe_import_torch()
    data = np.load(config.sequence_dataset_path)
    train_X, val_X, test_X = prepare_neural_arrays(data["train_X"], data["validation_X"], data["test_X"])
    train_y = data["train_y"].astype(np.float32)
    val_y = data["validation_y"].astype(np.float32)
    test_y = data["test_y"].astype(np.float32)

    class RNNClassifier(nn.Module):
        def __init__(self, input_size: int) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=48, batch_first=True)
            self.head = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

        def forward(self, inputs: Any) -> Any:
            outputs, _ = self.gru(inputs)
            return self.head(outputs[:, -1, :]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNClassifier(train_X.shape[-1]).to(device)
    pos_weight = torch.tensor([(train_y == 0).sum() / max(1, (train_y == 1).sum())], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.neural_learning_rate)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=config.neural_batch_size,
        shuffle=True,
    )
    for _ in range(config.neural_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    def infer(array: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(array, dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(tensor)).cpu().numpy()
        return scores

    payload = {
        "family": "neural_sequence",
        "artifact_path": str(config.outputs_dir / "best_model.pt"),
        "state_dict": model.state_dict(),
        "input_size": train_X.shape[-1],
        "normalization_mean": train_X.mean(axis=0).tolist(),
        "normalization_std": train_X.std(axis=0).tolist(),
    }
    return infer(val_X), infer(test_X), payload


def train_cnn_model(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    torch, nn, DataLoader, TensorDataset = maybe_import_torch()
    data = np.load(config.spatial_dataset_path)
    train_X, val_X, test_X = prepare_neural_arrays(data["train_X"], data["validation_X"], data["test_X"])
    train_y = data["train_y"].astype(np.float32)
    val_y = data["validation_y"].astype(np.float32)
    test_y = data["test_y"].astype(np.float32)

    class CNNClassifier(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, inputs: Any) -> Any:
            encoded = self.network(inputs).flatten(start_dim=1)
            return self.head(encoded).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(train_X.shape[1]).to(device)
    pos_weight = torch.tensor([(train_y == 0).sum() / max(1, (train_y == 1).sum())], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.neural_learning_rate)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(train_X, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32),
        ),
        batch_size=config.neural_batch_size,
        shuffle=True,
    )
    for _ in range(config.neural_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    def infer(array: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(array, dtype=torch.float32, device=device)
            scores = torch.sigmoid(model(tensor)).cpu().numpy()
        return scores

    payload = {
        "family": "neural_spatial",
        "artifact_path": str(config.outputs_dir / "best_model.pt"),
        "state_dict": model.state_dict(),
        "channels": train_X.shape[1],
        "normalization_mean": train_X.mean(axis=0).tolist(),
        "normalization_std": train_X.std(axis=0).tolist(),
    }
    return infer(val_X), infer(test_X), payload


def resolve_model_ids_for_track(config: PipelineConfig, track: TrackSpec, model_ids: list[str] | None = None) -> list[str]:
    requested = list(config.benchmark_models) if model_ids is None else list(model_ids)
    if track.track_id == "severe":
        filtered = [model_id for model_id in requested if model_id not in NEURAL_MODEL_IDS]
        skipped = sorted(set(requested) - set(filtered))
        if skipped:
            LOGGER.info("Skipping neural models for severe track: %s", ", ".join(skipped))
        return filtered
    return requested


def select_better_candidate(current: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current_val = current["validation"]
    best_val = best["validation"]
    current_tuple = (
        -1 if current_val["top_5pct_capture"] is None else current_val["top_5pct_capture"],
        -1 if current_val["average_precision"] is None else current_val["average_precision"],
        -1 if current_val["roc_auc"] is None else current_val["roc_auc"],
    )
    best_tuple = (
        -1 if best_val["top_5pct_capture"] is None else best_val["top_5pct_capture"],
        -1 if best_val["average_precision"] is None else best_val["average_precision"],
        -1 if best_val["roc_auc"] is None else best_val["roc_auc"],
    )
    return current_tuple > best_tuple


def save_best_artifact(best_payload: dict[str, Any], artifact_path: Path) -> None:
    if best_payload["family"] == "tabular":
        with artifact_path.open("wb") as handle:
            pickle.dump(best_payload["model"], handle)
    else:
        torch, _, _, _ = maybe_import_torch()
        torch.save(best_payload["state_dict"], best_payload["artifact_path"])


def summarize_hotspot_explanations(
    hotspots: pd.DataFrame,
    explain_frame: pd.DataFrame,
    shap_matrix: np.ndarray,
    feature_names: list[str],
    track_id: str,
) -> pd.DataFrame:
    signed = pd.DataFrame(shap_matrix, columns=feature_names)
    signed["cell_id"] = explain_frame["cell_id"].to_numpy()
    absolute = signed[feature_names].abs()
    positive = signed[feature_names].clip(lower=0)

    grouped_signed = signed.groupby("cell_id")[feature_names].mean()
    grouped_absolute = absolute.groupby(signed["cell_id"]).mean()
    grouped_positive = positive.groupby(signed["cell_id"]).mean()
    rows: list[dict[str, Any]] = []

    for hotspot in hotspots.itertuples(index=False):
        cell_id = hotspot.cell_id
        if cell_id not in grouped_signed.index:
            continue
        positive_scores = grouped_positive.loc[cell_id]
        ranking_scores = positive_scores if float(positive_scores.max()) > 0 else grouped_absolute.loc[cell_id]
        top_features = [name for name in ranking_scores.nlargest(3).index if float(ranking_scores[name]) > 0]
        row = {
            "cell_id": cell_id,
            "target_track": track_id,
            "hotspot_score": float(hotspot.mean_probability),
            "bucket_count": int(hotspot.bucket_count),
            "observed_positive_buckets": int(hotspot.observed_positive_buckets),
            "top_features": ", ".join(top_features),
            "cause_summary": compose_cause_summary(top_features),
        }
        for index in range(3):
            feature_name = top_features[index] if index < len(top_features) else ""
            row[f"feature_{index + 1}"] = feature_name
            row[f"feature_{index + 1}_display_name"] = feature_display_name(feature_name) if feature_name else ""
            row[f"feature_{index + 1}_mean_shap"] = (
                float(grouped_signed.loc[cell_id, feature_name]) if feature_name else 0.0
            )
            row[f"feature_{index + 1}_mean_abs_shap"] = (
                float(grouped_absolute.loc[cell_id, feature_name]) if feature_name else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def generate_shap_outputs(
    config: PipelineConfig,
    track: TrackSpec,
    metadata: dict[str, Any],
    hotspots: pd.DataFrame,
) -> dict[str, Any] | None:
    explainability = metadata.get("explainability", {})
    if not isinstance(explainability, dict) or not explainability.get("available"):
        return None
    if not track.explain_model_path.exists():
        LOGGER.warning("Missing explain model artifact for %s track: %s", track.track_id, track.explain_model_path)
        return None

    try:
        import shap
    except ImportError:
        LOGGER.warning("The 'shap' package is not installed. Skipping explanation export for %s track.", track.track_id)
        return None

    sample = pd.read_parquet(config.sample_path)
    train_df = sample[sample["split"] == "train"].copy()
    test_df = sample[sample["split"] == "test"].copy()
    if hotspots.empty or test_df.empty:
        return None

    explain_cells = set(hotspots["cell_id"].tolist())
    explain_frame = test_df[test_df["cell_id"].isin(explain_cells)].copy().sort_values("row_id").reset_index(drop=True)
    if explain_frame.empty:
        return None

    explain_model_id = str(explainability.get("model_id", ""))
    if explain_model_id not in TREE_TABULAR_MODEL_IDS:
        LOGGER.warning(
            "Skipping SHAP for track=%s explain_model=%s because it is not a supported tree explainer target",
            track.track_id,
            explain_model_id or "unknown",
        )
        return {
            "status": "failed",
            "explain_model_id": explain_model_id,
            "error": f"unsupported_explainer_model:{explain_model_id or 'unknown'}",
        }

    feature_names = list(config.tabular_feature_columns)
    background = train_df[feature_names].sample(
        n=min(200, len(train_df)),
        random_state=config.random_seed,
    )
    with track.explain_model_path.open("rb") as handle:
        explain_model = pickle.load(handle)

    X_explain = explain_frame[feature_names]
    background_array = background.to_numpy(dtype=np.float64, copy=True)
    explain_array = X_explain.to_numpy(dtype=np.float64, copy=True)

    try:
        explainer = shap.TreeExplainer(explain_model, data=background_array)
        explanation = explainer.shap_values(explain_array)
        shap_matrix = normalize_shap_matrix(explanation)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, X_explain, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(track.shap_summary_path, dpi=200, bbox_inches="tight")
        plt.close()

        hotspot_explanations = summarize_hotspot_explanations(
            hotspots=hotspots,
            explain_frame=explain_frame,
            shap_matrix=shap_matrix,
            feature_names=feature_names,
            track_id=track.track_id,
        )
        hotspot_explanations.to_csv(track.hotspot_explanations_path, index=False)
        return {
            "status": "ok",
            "explain_model_id": explain_model_id,
            "rows": int(len(hotspot_explanations)),
            "shap_summary_path": str(track.shap_summary_path),
            "hotspot_explanations_path": str(track.hotspot_explanations_path),
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "SHAP export failed for track=%s explain_model=%s error_type=%s",
            track.track_id,
            explain_model_id or "unknown",
            exc.__class__.__name__,
        )
        plt.close("all")
        return {
            "status": "failed",
            "explain_model_id": explain_model_id,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def benchmark_track(
    config: PipelineConfig,
    track: TrackSpec,
    sample: pd.DataFrame,
    model_ids: list[str] | None = None,
) -> Path:
    ensure_track_directories(track)
    track_model_ids = resolve_model_ids_for_track(config, track, model_ids=model_ids)
    if not track_model_ids:
        raise ValueError(f"No eligible benchmark models remain for track '{track.track_id}'.")

    train_df = sample[sample["split"] == "train"].copy()
    validation_df = sample[sample["split"] == "validation"].copy()
    test_df = sample[sample["split"] == "test"].copy()
    X_train = train_df[list(config.tabular_feature_columns)]
    X_val = validation_df[list(config.tabular_feature_columns)]
    X_test = test_df[list(config.tabular_feature_columns)]
    y_train = train_df[track.target_column]
    y_val = validation_df[track.target_column]
    y_test = test_df[track.target_column]

    results: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}
    best_candidate: dict[str, Any] | None = None
    best_tree_candidate: dict[str, Any] | None = None

    for model_id in track_model_ids:
        started = time.perf_counter()
        try:
            if model_id in NEURAL_MODEL_IDS:
                if model_id == "rnn_sequence":
                    val_scores, test_scores, artifact_payload = train_rnn_model(config)
                    val_row_ids = np.load(config.sequence_dataset_path)["validation_row_id"]
                    test_row_ids = np.load(config.sequence_dataset_path)["test_row_id"]
                else:
                    val_scores, test_scores, artifact_payload = train_cnn_model(config)
                    val_row_ids = np.load(config.spatial_dataset_path)["validation_row_id"]
                    test_row_ids = np.load(config.spatial_dataset_path)["test_row_id"]
                val_frame = validation_df.set_index("row_id").loc[val_row_ids].reset_index()
                test_frame = test_df.set_index("row_id").loc[test_row_ids].reset_index()
                validation_metrics = evaluate_predictions(val_frame[track.target_column], val_scores, config)
                test_metrics = evaluate_predictions(test_frame[track.target_column], test_scores, config)
                test_predictions = build_prediction_frame(test_frame, test_scores, track.target_column, track.track_id)
                payload = {"family": artifact_payload["family"], **artifact_payload}
            else:
                model = build_tabular_model(model_id, y_train, config)
                model.fit(X_train, y_train)
                val_scores = predict_scores(model, X_val)
                test_scores = predict_scores(model, X_test)
                validation_metrics = evaluate_predictions(y_val, val_scores, config)
                test_metrics = evaluate_predictions(y_test, test_scores, config)
                test_predictions = build_prediction_frame(test_df, test_scores, track.target_column, track.track_id)
                payload = {"family": "tabular", "model": model}

            runtime_seconds = round(time.perf_counter() - started, 3)
            result = {
                "target_track": track.track_id,
                "target_column": track.target_column,
                "model_id": model_id,
                "family": payload["family"],
                "status": "ok",
                "runtime_seconds": runtime_seconds,
                "feature_set_id": "tabular_v2",
                "validation": validation_metrics,
                "test": test_metrics,
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_average_precision": validation_metrics["average_precision"],
                "validation_top_5pct_capture": validation_metrics["top_5pct_capture"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_average_precision": test_metrics["average_precision"],
                "test_top_5pct_capture": test_metrics["top_5pct_capture"],
            }
            results.append(result)
            raw_results[model_id] = result
            candidate = {
                **result,
                "payload": payload,
                "test_predictions": test_predictions,
            }
            if select_better_candidate(candidate, best_candidate):
                best_candidate = candidate
            if model_id in TREE_TABULAR_MODEL_IDS and payload["family"] == "tabular":
                if select_better_candidate(candidate, best_tree_candidate):
                    best_tree_candidate = candidate
        except Exception as exc:  # noqa: BLE001
            runtime_seconds = round(time.perf_counter() - started, 3)
            failure = {
                "target_track": track.track_id,
                "target_column": track.target_column,
                "model_id": model_id,
                "family": "unknown",
                "status": "failed",
                "runtime_seconds": runtime_seconds,
                "feature_set_id": "tabular_v2",
                "error": str(exc),
            }
            results.append(failure)
            raw_results[model_id] = failure
            LOGGER.warning("Model %s failed during %s benchmark: %s", model_id, track.track_id, exc)

    leaderboard = pd.DataFrame(results)
    leaderboard.to_csv(track.leaderboard_path, index=False)
    benchmark_summary = {
        "target_track": track.track_id,
        "target_column": track.target_column,
        "models": raw_results,
    }
    track.benchmark_metrics_path.write_text(json.dumps(benchmark_summary, indent=2) + "\n")
    if best_candidate is None:
        raise ValueError(f"No benchmark model completed successfully for track '{track.track_id}'.")

    save_best_artifact(best_candidate["payload"], track.model_path)
    explainability: dict[str, Any] = {"available": False}
    if best_tree_candidate is not None:
        save_best_artifact(best_tree_candidate["payload"], track.explain_model_path)
        explainability = {
            "available": True,
            "model_id": best_tree_candidate["model_id"],
            "family": best_tree_candidate["family"],
            "source": "promoted_model" if best_tree_candidate["model_id"] == best_candidate["model_id"] else "best_tree_fallback",
        }

    best_metadata = {
        "target_track": track.track_id,
        "target_column": track.target_column,
        "model_id": best_candidate["model_id"],
        "family": best_candidate["family"],
        "selection_rule": "validation top_5pct_capture -> average_precision -> roc_auc",
        "validation": best_candidate["validation"],
        "test": best_candidate["test"],
        "feature_set_id": best_candidate["feature_set_id"],
        "explainability": explainability,
    }
    track.best_model_metadata_path.write_text(json.dumps(best_metadata, indent=2) + "\n")
    best_candidate["test_predictions"].to_csv(track.best_model_predictions_path, index=False)
    LOGGER.info("Benchmark complete for %s track. Best model: %s", track.track_id, best_candidate["model_id"])
    evaluate_best_track(config, track)
    plot_benchmark_summary(track)
    return track.leaderboard_path


def benchmark(
    config: PipelineConfig,
    model_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> Path:
    ensure_directories(config)
    sample = pd.read_parquet(config.sample_path)
    if sample.empty:
        raise ValueError("Sample dataset is empty.")

    resolved_tracks = target_ids or list(TRACK_IDS)
    last_path = build_track_spec(config, resolved_tracks[0]).leaderboard_path
    for track_id in resolved_tracks:
        track = build_track_spec(config, track_id)
        last_path = benchmark_track(config, track, sample, model_ids=model_ids)
    write_track_comparison_summary(config, resolved_tracks)
    return last_path


def compute_fars_overlap(config: PipelineConfig, hotspots: pd.DataFrame) -> float | None:
    if not config.raw_fars_path.exists():
        return None
    usecols = ["YEAR", "COUNTYNAME", "LATITUDE", "LONGITUD"]
    fars = pd.read_csv(config.raw_fars_path, usecols=usecols, low_memory=False)
    fars["YEAR"] = pd.to_numeric(fars["YEAR"], errors="coerce")
    fars["LATITUDE"] = pd.to_numeric(fars["LATITUDE"], errors="coerce")
    fars["LONGITUD"] = pd.to_numeric(fars["LONGITUD"], errors="coerce")
    fars = fars.dropna(subset=["YEAR", "LATITUDE", "LONGITUD"])
    fars = fars[
        fars["LATITUDE"].between(-90, 90)
        & fars["LONGITUD"].between(-180, 180)
    ].copy()
    fars = fars[fars["YEAR"].between(config.min_year, config.max_year)].copy()
    county_norm = normalize_text(fars["COUNTYNAME"])
    mask = (
        county_norm.str.contains("bronx", na=False)
        | county_norm.str.contains("kings", na=False)
        | county_norm.str.contains("new york", na=False)
        | county_norm.str.contains("queens", na=False)
        | county_norm.str.contains("richmond", na=False)
    )
    fars = fars[mask].copy()
    if fars.empty:
        return None
    geometry = gpd.points_from_xy(fars["LONGITUD"], fars["LATITUDE"])
    gdf = gpd.GeoDataFrame(fars, geometry=geometry, crs=config.output_crs).to_crs(config.processing_crs)
    finite_mask = np.isfinite(gdf.geometry.x) & np.isfinite(gdf.geometry.y)
    gdf = gdf[finite_mask].copy()
    if gdf.empty:
        LOGGER.warning("No valid FARS rows remained after coordinate validation and projection")
        return None
    gdf["grid_x"] = np.floor(gdf.geometry.x / config.grid_size_meters).astype("int64")
    gdf["grid_y"] = np.floor(gdf.geometry.y / config.grid_size_meters).astype("int64")
    gdf["cell_id"] = gdf["grid_x"].astype(str) + "_" + gdf["grid_y"].astype(str)
    fars_cells = set(gdf["cell_id"].unique())
    hotspot_cells = set(hotspots["cell_id"].unique())
    if not fars_cells:
        return None
    return float(len(fars_cells & hotspot_cells) / len(fars_cells))


def evaluate_best_track(config: PipelineConfig, track: TrackSpec) -> Path:
    ensure_directories(config)
    ensure_track_directories(track)
    if not track.best_model_predictions_path.exists():
        raise FileNotFoundError(f"Missing best-model predictions file: {track.best_model_predictions_path}")
    predictions = pd.read_csv(track.best_model_predictions_path, parse_dates=["bucket_start"])
    predictions = predictions.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
    predictions.to_csv(track.predictions_path, index=False)
    hotspots = aggregate_hotspots(predictions, config)
    hotspots.to_csv(track.hotspots_path, index=False)

    metadata = json.loads(track.best_model_metadata_path.read_text()) if track.best_model_metadata_path.exists() else {}
    explanation_summary = generate_shap_outputs(config, track, metadata, hotspots)
    metrics = {
        "target_track": track.track_id,
        "target_column": track.target_column,
        "model_id": metadata.get("model_id"),
        "family": metadata.get("family"),
        "validation": metadata.get("validation"),
        "test": metadata.get("test"),
        "fars_hotspot_overlap": compute_fars_overlap(config, hotspots),
        "explainability": metadata.get("explainability"),
        "explanation_artifacts": explanation_summary,
    }
    track.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_reports(track)
    return track.predictions_path


def evaluate_best(config: PipelineConfig, target_ids: list[str] | None = None) -> Path:
    ensure_directories(config)
    resolved_tracks = target_ids or list(TRACK_IDS)
    last_path = build_track_spec(config, resolved_tracks[0]).predictions_path
    for track_id in resolved_tracks:
        track = build_track_spec(config, track_id)
        last_path = evaluate_best_track(config, track)
    write_track_comparison_summary(config, resolved_tracks)
    return last_path


def plot_metric_summary(metrics: dict[str, Any], track: TrackSpec) -> None:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        split_metrics = metrics.get(split, {})
        if not isinstance(split_metrics, dict):
            continue
        for metric_name in ("roc_auc", "average_precision", "top_5pct_capture"):
            value = split_metrics.get(metric_name)
            if value is not None:
                rows.append({"split": split.title(), "metric": metric_name, "value": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame["metric"] = frame["metric"].map(
        {
            "roc_auc": "ROC AUC",
            "average_precision": "Average Precision",
            "top_5pct_capture": "Top 5% Capture",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=frame, x="metric", y="value", hue="split", ax=ax, palette="Set2")
    ax.set_ylim(0, 1)
    ax.set_title(f"{display_track_name(track.track_id)} Track Metrics ({metrics.get('model_id', 'unknown')})")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(track.metrics_plot_path, dpi=200)
    plt.close(fig)


def plot_prediction_distribution(predictions: pd.DataFrame, track: TrackSpec) -> None:
    frame = predictions.copy()
    target_label = "Crash" if track.track_id == "all_crash" else "Severe Crash"
    frame["actual_label"] = frame["label"].map({0: f"No {target_label}", 1: target_label})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=frame,
        x="predicted_probability",
        hue="actual_label",
        bins=30,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title(f"{display_track_name(track.track_id)} Predicted Probability Distribution")
    ax.set_xlabel(f"Predicted {target_label.lower()} probability")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(track.score_distribution_plot_path, dpi=200)
    plt.close(fig)


def plot_hotspot_map(hotspots: pd.DataFrame, track: TrackSpec) -> None:
    frame = hotspots.head(min(250, len(hotspots))).copy()
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    scatter = ax.scatter(
        frame["centroid_lng"],
        frame["centroid_lat"],
        c=frame["mean_probability"],
        s=np.clip(frame["bucket_count"] * 10, 30, 300),
        cmap="magma",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.2,
    )
    ax.set_title(f"{display_track_name(track.track_id)} Top Hotspot Cells")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Mean predicted probability")
    fig.tight_layout()
    fig.savefig(track.hotspot_map_plot_path, dpi=200)
    plt.close(fig)


def plot_temporal_heatmap(predictions: pd.DataFrame, track: TrackSpec) -> None:
    day_labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    heatmap = (
        predictions.groupby(["day_of_week", "hour"])["predicted_probability"]
        .mean()
        .reset_index()
        .pivot(index="day_of_week", columns="hour", values="predicted_probability")
        .sort_index()
    )
    if heatmap.empty:
        return
    heatmap.index = [day_labels.get(index, str(index)) for index in heatmap.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(heatmap, cmap="YlOrRd", ax=ax)
    ax.set_title(f"{display_track_name(track.track_id)} Average Predicted Risk by Day and Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    fig.tight_layout()
    fig.savefig(track.temporal_heatmap_plot_path, dpi=200)
    plt.close(fig)


def plot_benchmark_summary(track: TrackSpec) -> Path | None:
    if not track.leaderboard_path.exists():
        return None
    leaderboard = pd.read_csv(track.leaderboard_path)
    leaderboard = leaderboard[leaderboard["status"] == "ok"].copy()
    if leaderboard.empty:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=leaderboard.sort_values("validation_top_5pct_capture", ascending=False),
        x="model_id",
        y="validation_top_5pct_capture",
        hue="family",
        dodge=False,
        ax=ax,
        palette="tab10",
    )
    ax.set_title(f"{display_track_name(track.track_id)} Leaderboard: Validation Top 5% Capture")
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation top 5% capture")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(track.benchmark_plot_path, dpi=200)
    plt.close(fig)
    return track.benchmark_plot_path


def plot_reports(track: TrackSpec) -> Path:
    ensure_track_directories(track)
    metrics = json.loads(track.metrics_path.read_text())
    predictions = pd.read_csv(track.predictions_path, parse_dates=["bucket_start"])
    hotspots = pd.read_csv(track.hotspots_path)
    sns.set_theme(style="whitegrid")
    plot_metric_summary(metrics, track)
    plot_prediction_distribution(predictions, track)
    plot_hotspot_map(hotspots, track)
    plot_temporal_heatmap(predictions, track)
    return track.metrics_plot_path


def write_track_comparison_summary(config: PipelineConfig, track_ids: list[str] | None = None) -> Path | None:
    comparison_rows: list[dict[str, Any]] = []
    resolved_tracks = track_ids or list(TRACK_IDS)
    for track_id in resolved_tracks:
        track = build_track_spec(config, track_id)
        if not track.metrics_path.exists() or not track.hotspots_path.exists():
            continue
        metrics = json.loads(track.metrics_path.read_text())
        hotspots = pd.read_csv(track.hotspots_path)
        comparison_rows.append(
            {
                "target_track": track_id,
                "model_id": metrics.get("model_id"),
                "test_roc_auc": (metrics.get("test") or {}).get("roc_auc"),
                "test_average_precision": (metrics.get("test") or {}).get("average_precision"),
                "test_top_5pct_capture": (metrics.get("test") or {}).get("top_5pct_capture"),
                "fars_hotspot_overlap": metrics.get("fars_hotspot_overlap"),
                "mean_top_hotspot_probability": (
                    float(hotspots["mean_probability"].head(min(25, len(hotspots))).mean()) if not hotspots.empty else None
                ),
            }
        )
    if len(comparison_rows) < 2:
        return None

    comparison_path = config.outputs_dir / "track_comparison.json"
    comparison_plot_path = config.outputs_dir / "track_comparison.png"
    comparison = {"tracks": comparison_rows}
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

    metric_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        track_name = display_track_name(row["target_track"])
        for metric_name in ("test_roc_auc", "test_average_precision", "test_top_5pct_capture", "fars_hotspot_overlap"):
            value = row.get(metric_name)
            if value is not None:
                metric_rows.append({"track": track_name, "metric": metric_name, "value": value})
        if row.get("mean_top_hotspot_probability") is not None:
            concentration_rows.append(
                {
                    "track": track_name,
                    "value": row["mean_top_hotspot_probability"],
                }
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame["metric"] = metric_frame["metric"].map(
        {
            "test_roc_auc": "Test ROC AUC",
            "test_average_precision": "Test Avg Precision",
            "test_top_5pct_capture": "Test Top 5% Capture",
            "fars_hotspot_overlap": "FARS Overlap",
        }
    )
    sns.barplot(data=metric_frame, x="metric", y="value", hue="track", ax=axes[0], palette="Set2")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Track Performance Comparison")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Score")

    concentration_frame = pd.DataFrame(concentration_rows)
    sns.barplot(data=concentration_frame, x="track", y="value", ax=axes[1], palette="rocket")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Top Hotspot Concentration")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean probability across top 25 hotspot cells")

    fig.tight_layout()
    fig.savefig(comparison_plot_path, dpi=200)
    plt.close(fig)
    return comparison_path


def parse_grid_coordinates(cell_ids: pd.Series) -> pd.DataFrame:
    parts = cell_ids.astype(str).str.split("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("Cell ids must use the '<grid_x>_<grid_y>' format.")
    coordinates = pd.DataFrame(
        {
            "grid_x": pd.to_numeric(parts[0], errors="coerce"),
            "grid_y": pd.to_numeric(parts[1], errors="coerce"),
        },
        index=cell_ids.index,
    )
    if coordinates.isna().any().any():
        invalid = cell_ids[coordinates.isna().any(axis=1)].head(5).tolist()
        raise ValueError(f"Encountered invalid cell ids while reconstructing polygons: {invalid}")
    return coordinates.astype("int64")


def build_cell_polygon_frame(frame: pd.DataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    polygon_frame = frame.copy()
    if "grid_x" not in polygon_frame.columns or "grid_y" not in polygon_frame.columns:
        coords = parse_grid_coordinates(polygon_frame["cell_id"])
        polygon_frame["grid_x"] = coords["grid_x"]
        polygon_frame["grid_y"] = coords["grid_y"]
    polygon_frame["grid_x"] = pd.to_numeric(polygon_frame["grid_x"], errors="coerce").astype("int64")
    polygon_frame["grid_y"] = pd.to_numeric(polygon_frame["grid_y"], errors="coerce").astype("int64")
    size = config.grid_size_meters
    geometry = [
        box(grid_x * size, grid_y * size, (grid_x + 1) * size, (grid_y + 1) * size)
        for grid_x, grid_y in zip(polygon_frame["grid_x"], polygon_frame["grid_y"])
    ]
    gdf = gpd.GeoDataFrame(polygon_frame, geometry=geometry, crs=config.processing_crs).to_crs(config.output_crs)
    if gdf.empty:
        raise ValueError("Cannot export an empty Kepler layer.")
    if not bool(gdf.geometry.is_valid.all()):
        raise ValueError("Generated invalid geometries while building Kepler export layers.")
    return gdf


def serialize_geojson(gdf: gpd.GeoDataFrame) -> str:
    serializable = gdf.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return json.dumps(json.loads(serializable.to_json()), indent=2) + "\n"


def estimate_kepler_zoom(bounds: np.ndarray) -> float:
    min_lng, min_lat, max_lng, max_lat = bounds
    lon_span = max(float(max_lng - min_lng), 1e-6)
    lat_span = max(float(max_lat - min_lat), 1e-6)
    zoom_lon = np.log2(360.0 / lon_span)
    zoom_lat = np.log2(170.0 / lat_span)
    return float(max(0.0, min(18.0, min(zoom_lon, zoom_lat) - 0.75)))


def build_kepler_config(
    prediction_cells: gpd.GeoDataFrame,
    hotspot_cells: gpd.GeoDataFrame,
) -> dict[str, Any]:
    bounds = hotspot_cells.total_bounds if not hotspot_cells.empty else prediction_cells.total_bounds
    center_lng = float((bounds[0] + bounds[2]) / 2.0)
    center_lat = float((bounds[1] + bounds[3]) / 2.0)
    timestamps = pd.to_datetime(prediction_cells["bucket_start"])
    min_time_ms = int(timestamps.min().timestamp() * 1000)
    max_time_ms = int(timestamps.max().timestamp() * 1000)
    return {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [
                    {
                        "id": "prediction_time_filter",
                        "dataId": ["prediction_cells"],
                        "name": ["bucket_start"],
                        "type": "timeRange",
                        "value": [min_time_ms, max_time_ms],
                        "enlarged": True,
                        "plotType": "histogram",
                        "animationWindow": "free",
                    }
                ],
                "layers": [
                    {
                        "id": "prediction_cells_layer",
                        "type": "geojson",
                        "config": {
                            "dataId": "prediction_cells",
                            "label": "Prediction Cells",
                            "color": [245, 158, 11],
                            "columns": {"geojson": "geometry"},
                            "isVisible": False,
                            "visConfig": {
                                "opacity": 0.65,
                                "strokeOpacity": 0.4,
                                "thickness": 0.5,
                                "strokeColor": [255, 255, 255],
                                "colorRange": {
                                    "name": "Warm",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#FFF7EC", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"],
                                },
                                "radius": 10,
                                "sizeRange": [0, 10],
                                "radiusRange": [0, 50],
                                "heightRange": [0, 500],
                                "elevationScale": 5,
                                "enableElevationZoomFactor": True,
                                "stroked": True,
                                "filled": True,
                                "enable3d": False,
                                "wireframe": False,
                            },
                            "hidden": False,
                            "textLabel": [],
                        },
                        "visualChannels": {
                            "colorField": {"name": "predicted_probability", "type": "real"},
                            "colorScale": "quantile",
                            "strokeColorField": None,
                            "strokeColorScale": "quantile",
                            "sizeField": None,
                            "sizeScale": "linear",
                            "heightField": None,
                            "heightScale": "linear",
                            "radiusField": None,
                            "radiusScale": "linear",
                        },
                    },
                    {
                        "id": "top_hotspot_cells_layer",
                        "type": "geojson",
                        "config": {
                            "dataId": "top_hotspot_cells",
                            "label": "Top Hotspot Cells",
                            "color": [220, 38, 38],
                            "columns": {"geojson": "geometry"},
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.7,
                                "strokeOpacity": 0.95,
                                "thickness": 1.4,
                                "strokeColor": [127, 29, 29],
                                "colorRange": {
                                    "name": "Reds",
                                    "type": "sequential",
                                    "category": "ColorBrewer",
                                    "colors": ["#FEE2E2", "#FCA5A5", "#EF4444", "#DC2626", "#7F1D1D"],
                                },
                                "radius": 10,
                                "sizeRange": [0, 10],
                                "radiusRange": [0, 50],
                                "heightRange": [0, 500],
                                "elevationScale": 5,
                                "enableElevationZoomFactor": True,
                                "stroked": True,
                                "filled": True,
                                "enable3d": False,
                                "wireframe": False,
                            },
                            "hidden": False,
                            "textLabel": [],
                        },
                        "visualChannels": {
                            "colorField": {"name": "mean_probability", "type": "real"},
                            "colorScale": "quantile",
                            "strokeColorField": None,
                            "strokeColorScale": "quantile",
                            "sizeField": None,
                            "sizeScale": "linear",
                            "heightField": None,
                            "heightScale": "linear",
                            "radiusField": None,
                            "radiusScale": "linear",
                        },
                    },
                ],
                "interactionConfig": {
                    "tooltip": {
                        "enabled": True,
                        "fieldsToShow": {
                            "prediction_cells": [
                                {"name": "cell_id", "format": None},
                                {"name": "bucket_start", "format": None},
                                {"name": "predicted_probability", "format": ".4f"},
                                {"name": "hour", "format": None},
                                {"name": "day_of_week", "format": None},
                            ],
                            "top_hotspot_cells": [
                                {"name": "cell_id", "format": None},
                                {"name": "mean_probability", "format": ".4f"},
                                {"name": "max_probability", "format": ".4f"},
                                {"name": "bucket_count", "format": None},
                                {"name": "observed_positive_buckets", "format": None},
                            ],
                        },
                    },
                    "brush": {"enabled": False, "size": 0.5},
                    "geocoder": {"enabled": False},
                    "coordinate": {"enabled": False},
                },
                "layerBlending": "normal",
                "splitMaps": [],
                "animationConfig": {"currentTime": min_time_ms, "speed": 1},
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": center_lat,
                "longitude": center_lng,
                "pitch": 0,
                "zoom": max(estimate_kepler_zoom(bounds), 10.5),
                "isSplit": False,
            },
            "mapStyle": {
                "styleType": "light",
                "topLayerGroups": {},
                "visibleLayerGroups": {
                    "label": True,
                    "road": True,
                    "border": False,
                    "building": True,
                    "water": True,
                    "land": True,
                    "3d building": False,
                },
                "threeDBuildingColor": [218.82023004728602, 223.47597962276103, 223.47597962276103],
            },
        },
    }


def write_kepler_fallback_html(track: TrackSpec) -> Path:
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>NYC Hotspots Kepler Export</title>
    <style>
      body {{
        font-family: Arial, sans-serif;
        margin: 2rem;
        color: #111827;
      }}
      code {{
        background: #f3f4f6;
        padding: 0.15rem 0.35rem;
        border-radius: 0.25rem;
      }}
    </style>
  </head>
  <body>
    <h1>NYC Hotspots Kepler Export</h1>
    <p>
      GeoJSON layers and the saved Kepler config were generated successfully, but the interactive
      HTML map could not be rendered because the <code>keplergl</code> Python package is not installed
      in this environment.
    </p>
    <p>Install the project requirements and rerun <code>python3 -m hotspots.pipeline export-kepler</code>.</p>
    <ul>
      <li><code>{track.kepler_prediction_cells_path}</code></li>
      <li><code>{track.kepler_hotspot_cells_path}</code></li>
      <li><code>{track.kepler_config_path}</code></li>
    </ul>
  </body>
</html>
"""
    track.kepler_html_path.write_text(html)
    return track.kepler_html_path


def render_kepler_html(
    prediction_cells: gpd.GeoDataFrame,
    hotspot_cells: gpd.GeoDataFrame,
    kepler_config: dict[str, Any],
    track: TrackSpec,
) -> Path:
    try:
        from keplergl import KeplerGl
    except ImportError:
        LOGGER.warning(
            "The 'keplergl' package is not installed. Wrote GeoJSON/config artifacts and a fallback HTML page instead."
        )
        return write_kepler_fallback_html(track)

    kepler_map = KeplerGl(height=800, data={}, config=kepler_config)
    kepler_map.add_data(data=prediction_cells, name="prediction_cells")
    kepler_map.add_data(data=hotspot_cells, name="top_hotspot_cells")
    kepler_map.save_to_html(file_name=str(track.kepler_html_path))
    return track.kepler_html_path


def render_leaflet_html(hotspot_cells: gpd.GeoDataFrame, track: TrackSpec) -> Path:
    try:
        import folium
    except ImportError:
        LOGGER.warning("The 'folium' package is not installed. Skipping Leaflet fallback map export.")
        return track.leaflet_html_path

    hotspot_center = [
        float(hotspot_cells.geometry.centroid.y.mean()),
        float(hotspot_cells.geometry.centroid.x.mean()),
    ]
    hotspot_map = folium.Map(location=hotspot_center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        probability = float(feature["properties"].get("mean_probability", 0.0))
        if probability >= 0.94:
            fill_color = "#7f1d1d"
        elif probability >= 0.90:
            fill_color = "#b91c1c"
        elif probability >= 0.85:
            fill_color = "#ef4444"
        else:
            fill_color = "#fca5a5"
        return {
            "fillColor": fill_color,
            "color": "#7f1d1d",
            "weight": 1,
            "fillOpacity": 0.7,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["cell_id", "mean_probability", "max_probability", "bucket_count", "observed_positive_buckets"],
        aliases=["Cell ID", "Mean probability", "Max probability", "Bucket count", "Observed positive buckets"],
        localize=True,
        sticky=False,
        labels=True,
    )
    popup = folium.GeoJsonPopup(
        fields=["cell_id", "mean_probability", "max_probability", "bucket_count", "observed_positive_buckets"],
        aliases=["Cell ID", "Mean probability", "Max probability", "Bucket count", "Observed positive buckets"],
        localize=True,
        labels=True,
    )
    folium.GeoJson(
        json.loads(hotspot_cells.to_json()),
        name="Top Hotspot Cells",
        style_function=style_function,
        tooltip=tooltip,
        popup=popup,
    ).add_to(hotspot_map)
    folium.LayerControl(collapsed=False).add_to(hotspot_map)
    hotspot_map.fit_bounds(
        [
            [float(hotspot_cells.total_bounds[1]), float(hotspot_cells.total_bounds[0])],
            [float(hotspot_cells.total_bounds[3]), float(hotspot_cells.total_bounds[2])],
        ]
    )
    hotspot_map.save(str(track.leaflet_html_path))
    return track.leaflet_html_path


def load_kepler_source_frames(config: PipelineConfig, track: TrackSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_source = track.predictions_path if track.predictions_path.exists() else track.best_model_predictions_path
    if not prediction_source.exists():
        raise FileNotFoundError(
            f"Missing prediction source for Kepler export. Expected {track.predictions_path} or {track.best_model_predictions_path}."
        )
    predictions = pd.read_csv(prediction_source, parse_dates=["bucket_start"])
    if predictions.empty:
        raise ValueError("Prediction source is empty; cannot export Kepler layers.")
    if track.hotspots_path.exists():
        hotspots = pd.read_csv(track.hotspots_path)
    else:
        ranked_predictions = predictions.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
        hotspots = aggregate_hotspots(ranked_predictions, config)
    if hotspots.empty:
        raise ValueError("Hotspot source is empty; cannot export Kepler layers.")
    return predictions, hotspots


def export_kepler(config: PipelineConfig, target_ids: list[str] | None = None) -> Path:
    ensure_directories(config)
    resolved_tracks = target_ids or list(TRACK_IDS)
    last_path = build_track_spec(config, resolved_tracks[0]).kepler_html_path
    for track_id in resolved_tracks:
        track = build_track_spec(config, track_id)
        ensure_track_directories(track)
        predictions, hotspots = load_kepler_source_frames(config, track)
        prediction_cells = build_cell_polygon_frame(
            predictions[
                [
                    "row_id",
                    "cell_id",
                    "bucket_start",
                    "year",
                    "month",
                    "day_of_week",
                    "hour",
                    "centroid_lat",
                    "centroid_lng",
                    "label",
                    "split",
                    "predicted_probability",
                    "predicted_label",
                ]
            ],
            config,
        )
        hotspot_cells = build_cell_polygon_frame(
            hotspots[
                [
                    "cell_id",
                    "centroid_lat",
                    "centroid_lng",
                    "mean_probability",
                    "max_probability",
                    "bucket_count",
                    "observed_positive_buckets",
                ]
            ],
            config,
        )

        track.kepler_prediction_cells_path.write_text(serialize_geojson(prediction_cells))
        track.kepler_hotspot_cells_path.write_text(serialize_geojson(hotspot_cells))
        kepler_config = build_kepler_config(prediction_cells, hotspot_cells)
        track.kepler_config_path.write_text(json.dumps(kepler_config, indent=2) + "\n")
        last_path = render_kepler_html(prediction_cells, hotspot_cells, kepler_config, track)
        render_leaflet_html(hotspot_cells, track)
        LOGGER.info(
            "Kepler artifacts written for %s track: prediction_rows=%s hotspot_rows=%s output_dir=%s",
            track.track_id,
            len(prediction_cells),
            len(hotspot_cells),
            track.kepler_dir,
        )
    return last_path


def run_stage(
    stage: str,
    config: PipelineConfig,
    model_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> Path:
    stage_map = {
        "prepare-data": prepare_data,
        "build-grid": build_grid,
        "build-features": build_tabular_features,
        "build-tabular-features": build_tabular_features,
        "sample": sample_training_data,
        "build-sequence-data": build_sequence_data,
        "build-spatial-data": build_spatial_data,
        "benchmark": lambda cfg: benchmark(cfg, model_ids=model_ids, target_ids=target_ids),
        "evaluate-best": lambda cfg: evaluate_best(cfg, target_ids=target_ids),
        "export-kepler": lambda cfg: export_kepler(cfg, target_ids=target_ids),
        "plot": lambda cfg: evaluate_best(cfg, target_ids=target_ids),
        "train": lambda cfg: benchmark(cfg, model_ids=model_ids, target_ids=target_ids),
    }
    if stage not in stage_map:
        raise ValueError(f"Unknown stage: {stage}")
    return stage_map[stage](config)


def run_all(
    config: PipelineConfig,
    model_ids: list[str] | None = None,
    target_ids: list[str] | None = None,
) -> None:
    resolved_tracks = target_ids or list(TRACK_IDS)
    prepare_data(config)
    build_grid(config)
    build_tabular_features(config)
    sample_training_data(config)
    requested_models = list(config.benchmark_models) if model_ids is None else model_ids
    needs_neural_data = "all_crash" in resolved_tracks and any(model_id in NEURAL_MODEL_IDS for model_id in requested_models)
    if needs_neural_data:
        build_sequence_data(config)
        build_spatial_data(config)
    benchmark(config, model_ids=model_ids, target_ids=resolved_tracks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NYC accident hotspot multi-model benchmark pipeline")
    parser.add_argument(
        "stage",
        choices=[
            "run-all",
            "prepare-data",
            "build-grid",
            "build-features",
            "build-tabular-features",
            "sample",
            "build-sequence-data",
            "build-spatial-data",
            "benchmark",
            "evaluate-best",
            "export-kepler",
            "plot",
            "train",
        ],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model ids to benchmark, e.g. logreg,xgboost,svm_rbf",
    )
    parser.add_argument(
        "--targets",
        help="Comma-separated track ids to run. Supported values: all_crash,severe. Defaults to both.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = PipelineConfig()
    model_ids = args.models.split(",") if args.models else None
    target_ids = resolve_track_ids(args.targets)
    if args.stage == "run-all":
        run_all(config, model_ids=model_ids, target_ids=target_ids)
    else:
        run_stage(args.stage, config, model_ids=model_ids, target_ids=target_ids)


if __name__ == "__main__":
    main()
