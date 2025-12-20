"""
mlops_src/update_artifacts.py

This script copies newly trained artifacts from:
    backend_artifacts/
to:
    docker_backend/artifacts/

It ensures:
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

# filenames expected in backend_artifacts
EXPECTED = [
    "latest_column_transformer.joblib",
    "xgb_flight_price_model.joblib"
]

# --- LOGGING SETUP ---
LOG_DIR = os.path.join(PROJECT_ROOT, "mlops_src", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "update_artifacts.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("update_artifacts")


def verify_sources_exist():
    """Ensure model + transformer exist in backend_artifacts"""
    missing = []

    for fname in EXPECTED:
        if not os.path.exists(os.path.join(SRC_DIR, fname)):
            missing.append(fname)

    if missing:
        logger.error(f"Missing artifact(s): {missing}")
        raise FileNotFoundError(f"Run training first → Missing: {missing}")


def clean_destination():
    """Remove old artifacts from docker_backend/artifacts"""
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR, exist_ok=True)
        return

    for file in os.listdir(DEST_DIR):
        path = os.path.join(DEST_DIR, file)
        if os.path.isfile(path):
            logger.info(f"Removing old artifact: {path}")
            os.remove(path)


def copy_artifacts():
    """Copy new artifacts to docker_backend/artifacts"""
    for fname in EXPECTED:
        src_path = os.path.join(SRC_DIR, fname)
        dest_path = os.path.join(DEST_DIR, fname)

        logger.info(f"Copying {src_path} → {dest_path}")
        shutil.copy2(src_path, dest_path)


def main():
    logger.info("===== UPDATE ARTIFACTS STARTED =====")

    if not os.path.exists(SRC_DIR):
        logger.error(f"backend_artifacts/ NOT FOUND → {SRC_DIR}")
        raise FileNotFoundError("backend_artifacts folder missing")

    verify_sources_exist()
    clean_destination()
    copy_artifacts()

    logger.info("Artifacts updated successfully!")
    logger.info("===== UPDATE ARTIFACTS COMPLETED =====")

    print("✅ Artifacts copied to docker_backend/artifacts")


if __name__ == "__main__":
    main()
