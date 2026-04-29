from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    package_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    raw_accidents_file: str = "US_Accidents_March23.csv"
    raw_nypd_file: str = "Motor_Vehicle_Collisions_-_Crashes.csv"
    raw_fars_file: str = "accident.csv"
    random_seed: int = 42
    chunksize: int = 250_000
    nypd_chunksize: int = 150_000
    grid_size_meters: int = 250
    historical_sample_fraction: float = 0.20
    positive_negative_ratio: int = 4
    negative_pool_ratio: int = 8
    validation_fraction: float = 0.20
    test_year: int = 2022
    min_year: int = 2020
    max_year: int = 2022
    top_hotspot_fraction: float = 0.05
    processing_crs: str = "EPSG:32618"
    output_crs: str = "EPSG:4326"
    neural_lookback: int = 8
    spatial_patch_radius: int = 1
    neural_epochs: int = 6
    neural_batch_size: int = 256
    neural_learning_rate: float = 1e-3
    benchmark_models: tuple[str, ...] = (
        "logreg",
        "svm_rbf",
        "bagging_tree",
        "random_forest",
        "extra_trees",
        "hist_gb",
        "xgboost",
        "rnn_sequence",
        "cnn_spatial",
    )
    tabular_feature_columns: tuple[str, ...] = (
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_rush",
        "is_night",
        "centroid_lat",
        "centroid_lng",
        "prior_cell_crash_count",
        "rolling_recent_crash_count",
        "prior_cell_severe_count",
        "prior_cell_hour_crash_count",
        "prior_cell_dow_hour_crash_count",
        "hours_since_last_cell_crash",
        "prior_avg_severity",
        "prior_night_rate",
        "historic_cell_crash_total",
        "historic_neighbor_crash_total",
        "historic_hotspot_density",
        "observed_neighbor_count",
        "nypd_prior_collision_count",
        "nypd_prior_injury_count",
        "nypd_prior_killed_count",
        "nypd_prior_aggressive_count",
        "nypd_prior_distracted_count",
        "nypd_prior_yield_count",
        "nypd_prior_speed_count",
        "nypd_prior_alcohol_count",
        "weather_rain_history",
        "weather_snow_history",
        "weather_fog_history",
        "borough_bronx",
        "borough_brooklyn",
        "borough_manhattan",
        "borough_queens",
        "borough_staten_island",
    )
    sequence_feature_columns: tuple[str, ...] = (
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_rush",
        "is_night",
        "prior_cell_crash_count",
        "rolling_recent_crash_count",
        "prior_cell_severe_count",
        "prior_cell_hour_crash_count",
        "prior_cell_dow_hour_crash_count",
        "hours_since_last_cell_crash",
        "prior_avg_severity",
        "prior_night_rate",
        "nypd_prior_collision_count",
        "nypd_prior_injury_count",
        "nypd_prior_killed_count",
        "weather_rain_history",
        "weather_snow_history",
        "weather_fog_history",
        "delta_hours",
    )
    spatial_patch_channels: tuple[str, ...] = (
        "prior_cell_crash_count",
        "rolling_recent_crash_count",
        "prior_cell_severe_count",
        "prior_cell_hour_crash_count",
        "nypd_prior_collision_count",
        "historic_cell_crash_total",
        "historic_neighbor_crash_total",
    )
    nypd_factor_categories: tuple[str, ...] = (
        "aggressive",
        "distracted",
        "yield",
        "speed",
        "alcohol",
    )
    nyc_counties: tuple[str, ...] = (
        "bronx",
        "kings",
        "new york",
        "queens",
        "richmond",
    )
    nyc_cities: tuple[str, ...] = (
        "bronx",
        "brooklyn",
        "manhattan",
        "new york",
        "queens",
        "staten island",
    )
    required_columns: tuple[str, ...] = (
        "State",
        "City",
        "County",
        "Start_Time",
        "Start_Lat",
        "Start_Lng",
        "Severity",
        "Sunrise_Sunset",
        "Weather_Condition",
    )

    @property
    def raw_dir(self) -> Path:
        return self.package_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.package_dir / "data" / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.package_dir / "outputs"

    @property
    def benchmark_dir(self) -> Path:
        return self.outputs_dir / "benchmark"

    @property
    def kepler_dir(self) -> Path:
        return self.outputs_dir / "kepler"

    @property
    def raw_accidents_path(self) -> Path:
        return self.raw_dir / self.raw_accidents_file

    @property
    def raw_nypd_path(self) -> Path:
        return self.raw_dir / self.raw_nypd_file

    @property
    def raw_fars_path(self) -> Path:
        return self.raw_dir / self.raw_fars_file

    @property
    def prepared_path(self) -> Path:
        return self.processed_dir / "prepared_accidents.parquet"

    @property
    def grid_path(self) -> Path:
        return self.processed_dir / "grid_accidents.parquet"

    @property
    def cell_catalog_path(self) -> Path:
        return self.processed_dir / "cell_catalog.parquet"

    @property
    def nypd_hourly_path(self) -> Path:
        return self.processed_dir / "nypd_hourly.parquet"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def feature_columns_path(self) -> Path:
        return self.outputs_dir / "feature_columns.json"

    @property
    def sample_path(self) -> Path:
        return self.processed_dir / "sample.parquet"

    @property
    def sequence_dataset_path(self) -> Path:
        return self.processed_dir / "sequence_data.npz"

    @property
    def sequence_metadata_path(self) -> Path:
        return self.processed_dir / "sequence_metadata.json"

    @property
    def spatial_dataset_path(self) -> Path:
        return self.processed_dir / "spatial_data.npz"

    @property
    def spatial_metadata_path(self) -> Path:
        return self.processed_dir / "spatial_metadata.json"

    @property
    def leaderboard_path(self) -> Path:
        return self.benchmark_dir / "leaderboard.csv"

    @property
    def benchmark_metrics_path(self) -> Path:
        return self.benchmark_dir / "metrics.json"

    @property
    def benchmark_plot_path(self) -> Path:
        return self.benchmark_dir / "leaderboard.png"

    @property
    def best_model_metadata_path(self) -> Path:
        return self.outputs_dir / "best_model.json"

    @property
    def best_model_predictions_path(self) -> Path:
        return self.benchmark_dir / "best_model_test_predictions.csv"

    @property
    def model_path(self) -> Path:
        return self.outputs_dir / "best_model.pkl"

    @property
    def metrics_path(self) -> Path:
        return self.outputs_dir / "metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.outputs_dir / "test_predictions.csv"

    @property
    def hotspots_path(self) -> Path:
        return self.outputs_dir / "top_hotspots.csv"

    @property
    def metrics_plot_path(self) -> Path:
        return self.outputs_dir / "metrics_summary.png"

    @property
    def score_distribution_plot_path(self) -> Path:
        return self.outputs_dir / "prediction_score_distribution.png"

    @property
    def hotspot_map_plot_path(self) -> Path:
        return self.outputs_dir / "top_hotspots_map.png"

    @property
    def temporal_heatmap_plot_path(self) -> Path:
        return self.outputs_dir / "hourly_risk_heatmap.png"

    @property
    def kepler_prediction_cells_path(self) -> Path:
        return self.kepler_dir / "prediction_cells.geojson"

    @property
    def kepler_hotspot_cells_path(self) -> Path:
        return self.kepler_dir / "top_hotspot_cells.geojson"

    @property
    def kepler_config_path(self) -> Path:
        return self.kepler_dir / "kepler_config.json"

    @property
    def kepler_html_path(self) -> Path:
        return self.kepler_dir / "nyc_hotspots_map.html"

    @property
    def leaflet_html_path(self) -> Path:
        return self.kepler_dir / "nyc_hotspots_leaflet.html"
