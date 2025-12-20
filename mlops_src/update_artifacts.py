"""
mlops_src/update_artifacts.py

This script copies newly trained artifacts from:
    backend_artifacts/
to:
    docker_backend/artifacts/

It ensures:
 - destination folder always exists
 - outdated old artifacts are removed
 - latest transformer + model exist
 - logs all steps for debugging

Used by CI/CD pipeline before Docker build.
"""

import os
import shutil
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(PROJECT_ROOT, "backend_artifacts")
DEST_DIR = os.path.join(PROJECT_ROOT, "docker_backend", "artifacts")

EXPECTED = [
    "latest_column_transformer.joblib",
    "xgb_flight_price_model.joblib"
]

# --- Logging ---
LOG_DIR = os.path.join(PROJECT_ROOT, "mlops_src", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "update_artifacts.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("update_artifacts")


def ensure_destination_exists():
    """Ensure docker_backend/artifacts folder exists"""
    if not os.path.exists(DEST_DIR):
        logger.info(f"Creating destination folder: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)


def verify_sources_exist():
    """Ensure trained artifacts exist"""
    missing = []

    for fname in EXPECTED:
        if not os.path.exists(os.path.join(SRC_DIR, fname)):
            missing.append(fname)

    if missing:
        logger.error(f"Missing artifact(s): {missing}")
        raise FileNotFoundError(
            f"Run training first → Missing artifact(s): {missing}"
        )


def clean_destination():
    """Remove outdated files"""
    for item in os.listdir(DEST_DIR):
        path = os.path.join(DEST_DIR, item)
        if os.path.isfile(path):
            logger.info(f"Removing old artifact: {path}")
            os.remove(path)


def copy_artifacts():
    """Copy freshly trained artifacts"""
    for fname in EXPECTED:
        shutil.copy2(
            os.path.join(SRC_DIR, fname),
            os.path.join(DEST_DIR, fname)
        )
        logger.info(f"Copied → {fname}")


def main():
    logger.info("===== UPDATE ARTIFACTS STARTED =====")

    if not os.path.exists(SRC_DIR):
        raise FileNotFoundError(f"backend_artifacts not found: {SRC_DIR}")

    # NEW STEP - Ensure DESTINATION exists before ANY action
    ensure_destination_exists()

    verify_sources_exist()
    clean_destination()
    copy_artifacts()

    logger.info("===== UPDATE ARTIFACTS COMPLETED =====")
    print("✅ Artifacts copied to docker_backend/artifacts")


if __name__ == "__main__":
    main()
