"""Data loading helpers. Update filenames once Kaggle files are inspected."""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_csv(filename: str) -> pd.DataFrame:
    """Load an original CSV without modifying it."""
    return pd.read_csv(RAW_DIR / filename)
