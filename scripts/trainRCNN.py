import os
import time
import argparse
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

try:
	import pycocotools.coco as cocoapi
	from pycocotools.cocoeval import COCOeval
	hasPycoco = True
except Exception:
	hasPycoco = False

class CocoFormatDataset(Dataset):
	"""Thin wrapper around torchvision.datasets.CocoDetection to return targets in expected dict format."""
	def __init__(self, root: str, annFile: str, transforms=None):
		self.root = root
		self.annFile = annFile
		self.transforms = transforms
		self.coco = torchvision.datasets.CocoDetection(root=root, annFile=annFile)
		# torchvision's CocoDetection returns (PIL image, annotations list)
		# annotations are COCO dicts; we must convert to expected target format.

	def __len__(self):
		return len(self.coco)

	def __getitem__(self, idx):
		img, ann = self.coco[idx]
		# convert annotations to tensors expected by detection models
		# filter out annotations with "iscrowd" or invalid boxes
		boxes = []
		labels = []
		areas = []
		isCrowd = []

		for obj in ann:
			if "bbox" not in obj:
				continue
			x, y, w, h = obj["bbox"]
			if w <= 0 or h <= 0:
				continue
			# COCO bbox: [x,y,width,height] -> convert to [x1,y1,x2,y2]
			boxes.append([x, y, x + w, y + h])
			labels.append(obj.get("category_id", 0))
			areas.append(obj.get("area", w * h))
			isCrowd.append(obj.get("iscrowd", 0))

		if len(boxes) == 0:
			# Faster R-CNN needs at least one box per image during training; create dummy with label 0 and tiny area
			boxes = [[0.0, 0.0, 1.0, 1.0]]
			labels = [0]
			areas = [1.0]
			isCrowd = [0]

		# convert to tensors
		target = {}
		target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
		target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
		target["image_id"] = torch.tensor([idx])
		target["area"] = torch.as_tensor(areas, dtype=torch.float32)
		target["iscrowd"] = torch.as_tensor(isCrowd, dtype=torch.int64)

		if self.transforms:
			img, target = self.transforms(img, target)

		# convert PIL image to tensor
		imgTensor = F.to_tensor(img)

		return imgTensor, target

class FasterRCNNTrainer:
	"""Trainer class encapsulates model, optimizer, training loop and evaluation."""
	def __init__(self, device: str = "cuda"):
		self.device = torch.device(device if torch.cuda.is_available() else "cpu")
		self.model = None
		self.optimizer = None
		self.lrScheduler = None

	def buildModel(self, numClasses: int, pretrainedBackbone: bool = True):
		"""
		Build Faster R-CNN model.
		numClasses: number of classes (including background class as 0 in torchvision, but pass actual count)
		"""
		# torchvision expects num_classes = number of classes + 1 (for background)
		self.model = fasterrcnn_resnet50_fpn(pretrained=pretrainedBackbone)
		# replace the classifier with appropriate number of classes
		inFeatures = self.model.roi_heads.box_predictor.cls_score.in_features
		self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(inFeatures, numClasses)
		self.model.to(self.device)

	def setupOptimizer(self, lr: float = 0.005, momentum: float = 0.9, weightDecay: float = 0.0005):
		params = [p for p in self.model.parameters() if p.requires_grad]
		self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weightDecay)
		# optional LR scheduler
		self.lrScheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)

	@staticmethod
	def collateFn(batch):
		return tuple(zip(*batch))

	def train(self,
			  trainLoader: DataLoader,
			  valLoader: DataLoader = None,
			  numEpochs: int = 10,
			  checkpointDir: str = "results/checkpoints"):
		"""Main training loop."""
		if not os.path.exists(checkpointDir):
			os.makedirs(checkpointDir, exist_ok=True)

		self.model.train()
		for epoch in range(numEpochs):
			epochStart = time.time()
			lossesAcc = 0.0
			for images, targets in trainLoader:
				images = list(img.to(self.device) for img in images)
				targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

				lossDict = self.model(images, targets)
				losses = sum(loss for loss in lossDict.values())

				self.optimizer.zero_grad()
				losses.backward()
				self.optimizer.step()

				lossesAcc += losses.item()

			# step LR scheduler
			if self.lrScheduler is not None:
				self.lrScheduler.step()

			avgLoss = lossesAcc / len(trainLoader)
			epochTime = time.time() - epochStart
			print(f"[Epoch {epoch+1}/{numEpochs}] avgLoss: {avgLoss:.4f} time: {epochTime:.1f}s")

			# save checkpoint
			checkpointPath = os.path.join(checkpointDir, f"fasterrcnn_epoch{epoch+1}.pth")
			torch.save(self.model.state_dict(), checkpointPath)
			print(f"✅ Checkpoint saved: {checkpointPath}")

			# optional validation / evaluation
			if valLoader is not None:
				self.evaluate(valLoader)

	def evaluate(self, dataLoader: DataLoader):
		"""
		Basic evaluation loop: run inference and print simple stats.
		For COCO-style metrics, pycocotools is required and you must assemble predictions into COCO format.
		"""
		self.model.eval()
		allPredictions = []
		allGroundTruths = []

		with torch.no_grad():
			for images, targets in dataLoader:
				images = list(img.to(self.device) for img in images)
				outputs = self.model(images)

				# collect outputs and targets for potential COCO evaluation
				for tgt, out in zip(targets, outputs):
					# convert tensors to cpu numpy for later formatting
					imageId = int(tgt["image_id"].item()) if "image_id" in tgt else -1
					boxes = out["boxes"].cpu().numpy()
					scores = out["scores"].cpu().numpy()
					labels = out["labels"].cpu().numpy()
					allPredictions.append({
						"image_id": imageId,
						"boxes": boxes,
						"scores": scores,
						"labels": labels
					})
					allGroundTruths.append(tgt)

		print(f"📊 Inference completed on {len(allPredictions)} images")

		# If pycocotools available and original COCO dataset object available, run COCOeval (left as an exercise)
		if hasPycoco:
			print("ℹ️ pycocotools available - you can compute COCO mAP by converting predictions to COCO format.")

		self.model.train()
		return allPredictions

def parseArgs():
	parser = argparse.ArgumentParser(description="Train Faster R-CNN on COCO-style dataset")
	parser.add_argument("--dataRoot", type=str, required=True, help="Root folder containing images")
	parser.add_argument("--annFile", type=str, required=True, help="Path to COCO train annotations JSON")
	parser.add_argument("--valAnnFile", type=str, required=False, help="Path to COCO val annotations JSON")
	parser.add_argument("--numClasses", type=int, required=True, help="Number of classes (including background if needed)")
	parser.add_argument("--epochs", type=int, default=12)
	parser.add_argument("--batchSize", type=int, default=4)
	parser.add_argument("--lr", type=float, default=0.005)
	parser.add_argument("--checkpointDir", type=str, default="results/checkpoints")
	return parser.parse_args()

def main():
	args = parseArgs()

	# dataset and dataloaders
	trainDataset = CocoFormatDataset(root=os.path.join(args.dataRoot, "images/train2017"),
									 annFile=args.annFile)
	valDataset = None
	if args.valAnnFile:
		valDataset = CocoFormatDataset(root=os.path.join(args.dataRoot, "images/val2017"),
									  annFile=args.valAnnFile)

	trainLoader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True,
							 num_workers=4, collate_fn=FasterRCNNTrainer.collateFn)
	valLoader = None
	if valDataset:
		valLoader = DataLoader(valDataset, batch_size=1, shuffle=False,
							   num_workers=4, collate_fn=FasterRCNNTrainer.collateFn)

	# build trainer and model
	trainer = FasterRCNNTrainer()
	trainer.buildModel(numClasses=args.numClasses, pretrainedBackbone=True)
	trainer.setupOptimizer(lr=args.lr)

	# start training
	trainer.train(trainLoader=trainLoader, valLoader=valLoader, numEpochs=args.epochs, checkpointDir=args.checkpointDir)

if __name__ == "__main__":
	main()

