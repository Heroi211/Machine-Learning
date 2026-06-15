"""Download mínimo do MovieLens (ml-latest-small) para o pipeline TC02."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def ensure_movielens_sample(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    ratings = raw_dir / "ratings.csv"
    if ratings.is_file():
        return ratings

    zip_path = raw_dir / "ml-latest-small.zip"
    urlretrieve(MOVIELENS_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("ratings.csv"):
                data = zf.read(name)
                ratings.write_bytes(data)
                return ratings
    raise FileNotFoundError("ratings.csv não encontrado dentro do zip MovieLens.")
