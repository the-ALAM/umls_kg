from dataclasses import dataclass
from typing import List


@dataclass
class DatabaseConfig:
    url: str = "ws://localhost:8000/rpc"
    namespace: str = "relevance_model_ns"
    database: str = "omop_db"
    username: str = "root"
    password: str = "rootoot"

@dataclass
class PathConfig:
    source_csv_path: str = "/home/alam/lkr/kg2/src.csv"
    target_csv_path: str = "/home/alam/lkr/kg2/HF_TA_DRGs.csv"
    output_csv_path: str = "/home/alam/lkr/kg2/data/path_analysis_results.csv"

@dataclass
class TraversalConfig:
    max_depth: int = 1
    therapeutic_area: str = "heart_failure"
    target_drg_codes: List[int] = None
    source_concepts: List[str] = None

@dataclass
class ConfigLoader:
    @staticmethod
    def load_config() -> dict:
        # Placeholder for actual config loading logic
        return {
            "database": DatabaseConfig(),
            "paths": PathConfig(),
            "traversal": TraversalConfig(
                target_drg_codes=[291, 292, 293],
                source_concepts=[
                    "Congestive heart failure",
                    "Left ventricular failure",
                    "Cardiac failure",
                    "Heart failure",
                    "Right heart failure",
                    "Systolic heart failure",
                    "Diastolic heart failure"
                ]
            )
        }

@dataclass
class ApplicationConfig:
    database: DatabaseConfig
    paths: PathConfig
    traversal: TraversalConfig

    @staticmethod
    def create_default() -> 'ApplicationConfig':
        config_data = ConfigLoader.load_config()
        return ApplicationConfig(
            database=config_data["database"],
            paths=config_data["paths"],
            traversal=config_data["traversal"]
        )
