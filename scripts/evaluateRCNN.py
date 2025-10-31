"""
evaluateFasterRCNN.py
---------------------
Evaluates a trained Faster R-CNN model using COCO-style datasets.
Performs inference, visualizes predictions, and computes mAP automatically.

Requirements:
	torch, torchvision, matplotlib, pycocotools
"""

import os
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm

try:
	from pycocotools.coco import COCO
	from pycocotools.cocoeval import COCOeval
	hasPycoco = True
except Exception:
	hasPycoco = False


class FasterRCNNEvaluator:
	"""Evaluates Faster R-CNN models and computes COCO-style mAP."""

	def __init__(self, modelPath, dataRoot, annFile, numClasses, device="cuda"):
		self.modelPath = modelPath
		self.dataRoot = dataRoot
		self.annFile = annFile
		self.numClasses = numClasses
		self.device = torch.device(device if torch.cuda.is_available() else "cpu")
		self.model = None
		self.dataset = None
		self.dataLoader = None
		self.resultsDir = "results/evaluation"
		os.makedirs(self.resultsDir, exist_ok=True)

	def buildModel(self):
		"""Load Faster R-CNN model with trained weights."""
		print("🔧 Loading Faster R-CNN model...")
		self.model = fasterrcnn_resnet50_fpn(pretrained=False)
		inFeatures = self.model.roi_heads.box_predictor.cls_score.in_features
		from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
		self.model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, self.numClasses)
		self.model.load_state_dict(torch.load(self.modelPath, map_location=self.device))
		self.model.to(self.device)
		self.model.eval()
		print("✅ Model loaded successfully.")

	def loadDataset(self):
		"""Load COCO-style dataset."""
		print("📂 Loading dataset for evaluation...")
		self.dataset = CocoDetection(root=self.dataRoot, annFile=self.annFile)
		self.dataLoader = DataLoader(self.dataset, batch_size=1, shuffle=False,
									 num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
		print(f"✅ Loaded {len(self.dataset)} images for evaluation.")

	def runInference(self, scoreThreshold=0.05):
		"""Run model inference and collect predictions."""
		print("🔍 Running inference...")
		predictions = []
		with torch.no_grad():
			for imgs, tgts in tqdm(self.dataLoader, desc="Evaluating"):
				img = imgs[0].to(self.device)
				imageId = int(tgts[0]["image_id"]) if "image_id" in tgts[0] else -1

				output = self.model([img])[0]
				boxes = output["boxes"].cpu().numpy()
				scores = output["scores"].cpu().numpy()
				labels = output["labels"].cpu().numpy()

				for box, score, label in zip(boxes, scores, labels):
					if score < scoreThreshold:
						continue
					x1, y1, x2, y2 = box
					width = x2 - x1
					height = y2 - y1
					pred = {
						"image_id": imageId,
						"category_id": int(label),
						"bbox": [float(x1), float(y1), float(width), float(height)],
						"score": float(score)
					}
					predictions.append(pred)

		predFile = os.path.join(self.resultsDir, "fasterrcnn_predictions.json")
		with open(predFile, "w") as f:
			json.dump(predictions, f)
		print(f"✅ Predictions saved: {predFile}")
		return predFile

	def computeCocoMetrics(self, predFile):
		"""Compute COCO mAP using pycocotools."""
		if not hasPycoco:
			print("⚠️ pycocotools not installed; skipping mAP computation.")
			return

		print("📈 Running COCO-style evaluation...")
		cocoGt = COCO(self.annFile)
		cocoDt = cocoGt.loadRes(predFile)
		cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
		cocoEval.evaluate()
		cocoEval.accumulate()
		cocoEval.summarize()
		print("✅ Evaluation complete.")
		return cocoEval.stats

	def visualizePrediction(self, imageIndex=0, scoreThreshold=0.5):
		"""Visualize sample prediction."""
		print(f"🖼️ Visualizing prediction for sample {imageIndex}...")
		img, _ = self.dataset[imageIndex]
		imgTensor = F.to_tensor(img).unsqueeze(0).to(self.device)

		with torch.no_grad():
			preds = self.model(imgTensor)[0]

		boxes = preds["boxes"].cpu().numpy()
		scores = preds["scores"].cpu().numpy()
		labels = preds["labels"].cpu().numpy()

		plt.imshow(img)
		for box, score, label in zip(boxes, scores, labels):
			if score < scoreThreshold:
				continue
			x1, y1, x2, y2 = box
			plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
											  fill=False, color="lime", linewidth=2))
			plt.text(x1, y1 - 5, f"ID:{label} {score:.2f}",
					 color="yellow", fontsize=8, weight="bold")
		plt.axis("off")
		plt.title("Faster R-CNN Detections")
		plt.show()

	def evaluatePipeline(self):
		"""Run full evaluation pipeline."""
		self.buildModel()
		self.loadDataset()
		predFile = self.runInference()
		self.computeCocoMetrics(predFile)
		self.visualizePrediction(imageIndex=3)


if __name__ == "__main__":
	evaluator = FasterRCNNEvaluator(
		modelPath="results/checkpoints/fasterrcnn_epoch10.pth",
		dataRoot="data/coco/images/val2017",
		annFile="data/coco/annotations/instances_val2017.json",
		numClasses=81  # 80 COCO classes + background
	)

	evaluator.evaluatePipeline()

