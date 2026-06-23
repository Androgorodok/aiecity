from pathlib import Path
import zipfile
import subprocess

DATASET = "smaranjitghose/corn-or-maize-leaf-disease-dataset"

DATA_DIR = Path("data")

DATA_DIR.mkdir(exist_ok=True)

print("Downloading dataset...")

subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(DATA_DIR),
    ],
    check=True,
)

zip_file = next(DATA_DIR.glob("*.zip"))

with zipfile.ZipFile(zip_file, "r") as z:
    z.extractall(DATA_DIR)

zip_file.unlink()

print("Done")
