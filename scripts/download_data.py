"""Download the Jigsaw Toxic Comment data from Kaggle into ./data.

Usage (from the project root):
    python scripts/download_data.py

Needs a Kaggle API key. Either:
  - put kaggle.json in %USERPROFILE%\\.kaggle\\ (Windows) or ~/.kaggle/ (Mac/Linux), or
  - put kaggle.json in the project root (it is git-ignored).
You must also click "I understand and accept" on the competition's Rules page once:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules
"""
import os
import sys
import zipfile
from pathlib import Path

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NEEDED = ["train.csv", "test.csv", "test_labels.csv"]


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if all((DATA / f).exists() for f in NEEDED):
        print("All files already in data/ - nothing to do.")
        return

    if (ROOT / "kaggle.json").exists() and "KAGGLE_CONFIG_DIR" not in os.environ:
        os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("The kaggle package isn't installed. Run: pip install -r requirements.txt")

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Kaggle login failed ({e}). Put kaggle.json in ~/.kaggle/ or in the project root.")

    print("Downloading from Kaggle...")
    try:
        api.competition_download_files(COMPETITION, path=str(DATA), quiet=False)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Download failed ({e}). Have you accepted the competition rules on Kaggle?")

    # The competition zip contains more zips (train.csv.zip, ...). Unpack until none are left.
    while True:
        zips = list(DATA.glob("*.zip"))
        if not zips:
            break
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(DATA)
            z.unlink()

    missing = [f for f in NEEDED if not (DATA / f).exists()]
    if missing:
        sys.exit(f"Downloaded, but still missing: {missing}")
    for f in NEEDED:
        print(f"  {f}: {(DATA / f).stat().st_size / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
