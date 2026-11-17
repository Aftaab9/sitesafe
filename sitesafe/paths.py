"""Project paths in one place. Everything is relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DATASET = DATA / "construction-ppe"       # unzipped original (images/{train,val}, labels/{train,val})
SPLITS = DATA / "splits"                   # train.txt / val.txt / test.txt image lists
SYNTH = DATA / "synthetic"                 # generated images + labels
OWN = DATA / "own_test"                    # my own 30 images
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"
WEB_PUBLIC = ROOT / "web" / "public"

SEED = 42
