import os
import yaml
import json
import glob

class DataPreparer:
	"""Prepares datasets and creates YAML configuration files for YOLOv8."""
	def __init__(self, baseDir="data"):
		self.baseDir = baseDir
		self.datasets = ["kitti", "coco"]
		print(f"Using baseDir: {os.path.abspath(self.baseDir)}")
		print(f"Current working directory: {os.getcwd()}")

	def setupDirectories(self):
		"""Ensure required dataset directories exist."""
		if not os.path.exists(self.baseDir):
			os.makedirs(self.baseDir, exist_ok=True)
			print(f"Created base directory: {self.baseDir}") # data/ directory

		for ds in self.datasets:
			# make sure coco and kitti directories exist
			# making sure that coco/labels/train2017 and coco/labels/val2017 exist too
			os.makedirs(f"{self.baseDir}/{ds}", exist_ok=True)
			if ds == "coco":
				os.makedirs(f"{self.baseDir}/{ds}/labels/train2017", exist_ok=True)
				os.makedirs(f"{self.baseDir}/{ds}/labels/val2017", exist_ok=True)
			print(f"Directory verified: {self.baseDir}/{ds}")

	def convertCocoToYolo(self, jsonPath, imgDir, labelDir, classNames):
		"""Convert COCO JSON annotations to YOLO format."""
		# Ensure that necessary paths exist
		if not os.path.exists(jsonPath):
			print(f"COCO annotation file not found: {jsonPath}")
			return
		if not os.path.exists(imgDir):
			print(f"Image directory not found: {imgDir}")
			return
		
		# Create label directory if it doesn't exist
		os.makedirs(labelDir, exist_ok=True)
		with open(jsonPath, "r") as f:
			cocoData = json.load(f)
		
		# Map COCO category IDs to YOLO class IDs
		catIdToYoloId = {cat["id"]: classNames.index(cat["name"]) for cat in cocoData["categories"] if cat["name"] in classNames}
		imgIdToFile = {img["id"]: img["file_name"] for img in cocoData["images"]}

		# Process each image and create corresponding YOLO label files
		for imgId in imgIdToFile:
			# Create label file path
			labelFile = os.path.join(labelDir, os.path.splitext(imgIdToFile[imgId])[0] + ".txt")
			annotations = [ann for ann in cocoData["annotations"] if ann["image_id"] == imgId and ann["category_id"] in catIdToYoloId]

			if not annotations:
				continue

			# Get image dimensions
			imgInfo = next(img for img in cocoData["images"] if img["id"] == imgId)
			imgWidth, imgHeight = imgInfo["width"], imgInfo["height"]

			# Write YOLO formatted annotations
			with open(labelFile, "w") as f:
				for ann in annotations:
					x, y, w, h = ann["bbox"]
					xCenter = (x + w / 2) / imgWidth
					yCenter = (y + h / 2) / imgHeight
					width = w / imgWidth
					height = h / imgHeight
					classId = catIdToYoloId[ann["category_id"]]
					f.write(f"{classId} {xCenter:.6f} {yCenter:.6f} {width:.6f} {height:.6f}\n")
			print(f"Converted labels for {imgIdToFile[imgId]} to {labelFile}")

	def createDatasetYaml(self, datasetName, trainPath, valPath, classNames):
		"""Generate YOLOv8-compatible YAML file."""
		# Create YAML content
		yamlData = {
			"path": self.baseDir.rstrip("/"),
			"train": trainPath,
			"val": valPath,
			"names": classNames
		}

		# Write YAML file
		yamlPath = os.path.join(self.baseDir, f"{datasetName}.yaml")
		absYamlPath = os.path.abspath(yamlPath)
		try:
			# Check write permissions
			if not os.access(self.baseDir, os.W_OK):
				raise PermissionError(f"No write permission for {self.baseDir}")
			# Write YAML file
			with open(yamlPath, "w") as f:
				yaml.dump(yamlData, f, sort_keys=False)
			# Verify file creation
			if os.path.exists(yamlPath):
				print(f"Created {datasetName}.yaml at {absYamlPath}")
				with open(yamlPath, "r") as f:
					print(f"Content of {datasetName}.yaml:\n{f.read()}")
			else:
				print(f"Failed to verify {datasetName}.yaml at {absYamlPath}")
		except Exception as e:
			print(f"Error creating {datasetName}.yaml: {str(e)}")

if __name__ == "__main__":
	# Ensure script is run from the expected root directory
	expectedRoot = "/Users/jasnbone/Documents/Classes/cecs385/roadDetectionProject"
	if os.getcwd() != expectedRoot:
		print(f"Warning: Script should be run from {expectedRoot}, but current directory is {os.getcwd()}")
		print(f"Run: cd {expectedRoot} && python scripts/dataPrep.py")

	# Initialize DataPreparer and setup directories
	preparer = DataPreparer()
	preparer.setupDirectories()

	# Convert COCO annotations to YOLO format using predefined function
	cocoClassNames = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign"]
	# Convert training annotations
	preparer.convertCocoToYolo(
		jsonPath=f"{preparer.baseDir}/coco/annotations/instances_train2017.json",
		imgDir=f"{preparer.baseDir}/coco/train2017",
		labelDir=f"{preparer.baseDir}/coco/labels/train2017",
		classNames=cocoClassNames
	)
	# Convert validation annotations
	preparer.convertCocoToYolo(
		jsonPath=f"{preparer.baseDir}/coco/annotations/instances_val2017.json",
		imgDir=f"{preparer.baseDir}/coco/val2017",
		labelDir=f"{preparer.baseDir}/coco/labels/val2017",
		classNames=cocoClassNames
	)

	# Create dataset YAML files for KITTI and COCO
	preparer.createDatasetYaml(
		datasetName="kitti",
		trainPath="kitti/data_object_image_2/training/image_2",
		valPath="kitti/data_object_image_2/testing/image_2",
		classNames=["Car", "Pedestrian", "Cyclist"]
	)
	preparer.createDatasetYaml(
		datasetName="coco",
		trainPath="coco/train2017",
		valPath="coco/val2017",
		classNames=cocoClassNames
	)
