import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()

# Add src to Python's module search path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Check whether Python can locate the module
spec = importlib.util.find_spec("feature_engineering")
print("Module found:", spec is not None)

if spec is not None:
    print("Module location:", spec.origin)

# Try importing the two functions
from feature_engineering import build_features, check_feature_integrity

print("Both functions imported successfully!")