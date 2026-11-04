"""Train models, evaluate on an internal holdout, and predict Kaggle test.csv."""
from pathlib import Path
from src.modeling import train_and_save

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    train_and_save(ROOT)
