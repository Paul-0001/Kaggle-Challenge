from pathlib import Path

def test_expected_directories_exist():
    root = Path(__file__).resolve().parents[1]
    for directory in ("data/raw", "data/processed", "src/models", "src/preprocessing", "scripts"):
        assert (root / directory).is_dir()
