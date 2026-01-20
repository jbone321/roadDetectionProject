import json
import shutil
import os
from pathlib import Path
import zipfile

# -----------------------------
# Paths (adjust as needed)
# -----------------------------
fullTrainDir = "data/coco/train2017"          # full COCO train2017 folder on Mac
fullAnn = "data/coco/annotations/instances_train2017_subset.json"  # your subset JSON
outputDir = "data/cocoSubset"               # where subset folder will go
zipName = "cocoSubset.zip"                      # output zip file

# -----------------------------
# Create output folder
# -----------------------------
trainOut = os.path.join(outputDir, "train2017Subset")
annOut = os.path.join(outputDir, "annotations")
os.makedirs(trainOut, exist_ok=True)
os.makedirs(annOut, exist_ok=True)

# -----------------------------
# Load subset JSON and get filenames
# -----------------------------
with open(fullAnn, "r") as f:
    data = json.load(f)

files = [img["file_name"] for img in data["images"]]

# -----------------------------
# Copy images
# -----------------------------
print(f"Copying {len(files)} images...")
for fName in files:
    srcPath = os.path.join(fullTrainDir, fName)
    dstPath = os.path.join(trainOut, fName)
    if os.path.exists(srcPath):
        shutil.copy2(srcPath, dstPath)

# -----------------------------
# Copy JSON
# -----------------------------
shutil.copy2(fullAnn, os.path.join(annOut, "instances_train2017_subset.json"))

# -----------------------------
# Zip everything
# -----------------------------
print("Zipping subset...")
with zipfile.ZipFile(os.path.join(outputDir, zipName), 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(outputDir):
        for file in files:
            if file == zipName:
                continue  # skip the zip itself
            filePath = os.path.join(root, file)
            zipf.write(filePath, os.path.relpath(filePath, outputDir))

print(f"Subset ready: {os.path.join(outputDir, zipName)}")

