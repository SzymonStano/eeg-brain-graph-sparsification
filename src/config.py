from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "chb-mit"
PROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

COMMON_CHANNELS = [
    'C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 
    'FP1-F3', 'FP1-F7', 'FP2-F4', 'FP2-F8', 'FT10-T8', 'FT9-FT10', 
    'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P7-T7', 'P8-O2', 'T7-FT9', 
    'T7-P7', 'T8-P8'
]