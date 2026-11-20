from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Data Directories
RAW_DIR = ROOT / "data" / "raw"
RAW_REAL_DIR = RAW_DIR / "real_sources"
PROCESSED_DIR = ROOT / "data" / "processed"
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
METADATA_DIR = ROOT / "data" / "metadata"

# Production Real Datasets
YIELD_DATA = PROCESSED_DIR / "clean_apy_yield.csv"
SOIL_HEALTH_DATA = PROCESSED_DIR / "clean_soil_fertility.csv"
DISTRICT_RAINFALL_DATA = PROCESSED_DIR / "clean_district_rainfall.csv"
MANDI_PRICES_DATA = PROCESSED_DIR / "clean_mandi_prices.csv"
STATE_DISTRICT_MAP_PATH = PROCESSED_DIR / "state_district_hierarchy.json"

# Knowledge Bases
KNOWLEDGE_PATH = KNOWLEDGE_DIR / "schemes.json"
STATE_SCHEMES_PATH = KNOWLEDGE_DIR / "state_schemes.json"
STATE_CROP_CATALOG_PATH = KNOWLEDGE_DIR / "state_crop_catalog.csv"

# Output Directories
MODEL_DIR = ROOT / "models"
MODELS_DIR = MODEL_DIR
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "outputs"

# Experiment Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TOP_N_FEATURES = 60

