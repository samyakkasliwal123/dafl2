import os
import zipfile
import requests
from tqdm import tqdm

URL = "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip"
OUT_DIR = "./data"
ZIP_PATH = os.path.join(OUT_DIR, "VisDrone2019-DET-train.zip")
EXTRACT_DIR = os.path.join(OUT_DIR, "VisDrone2019-DET-train")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    print("Downloading VisDrone2019 training data...")
    response = requests.get(URL, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(ZIP_PATH, 'wb') as file, tqdm(
        desc=ZIP_PATH,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
else:
    print("Dataset zip already downloaded.")

print("Extracting data...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
print("Done.")