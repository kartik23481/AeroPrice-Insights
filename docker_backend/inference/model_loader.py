import os
import joblib

# Required for unpickling
from utils.feature_utils import (
    is_north,
    find_part_of_month,
    part_of_day,
    make_month_object,
    remove_duration,
    have_info,
    duration_category
)
import utils.rbf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = /app/docker_backend

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

TRANSFORMER_PATH = os.path.join(
    ARTIFACTS_DIR,
    "latest_column_transformer.joblib"
)

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "xgb_flight_price_model.joblib"
)

COLUMN_TRANSFORMER = joblib.load(TRANSFORMER_PATH)
MODEL = joblib.load(MODEL_PATH)
