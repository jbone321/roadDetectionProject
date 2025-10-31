"""
trainYOLOv8.py
---------------
Defines YOLOv8 training workflow using OOP principles.
"""

from ultralytics import YOLO

class YOLOTrainer:
	"""Trains YOLOv8 models on COCO and KITTI datasets."""
	def __init__(self, baseModel="yolov8n.pt"):
		self.baseModel = baseModel

	def trainModel(self, dataYaml, epochs, batch, modelName):
		"""Train YOLOv8 on the specified dataset."""
		model = YOLO(self.baseModel)
		results = model.train(
			data=dataYaml,
			epochs=epochs,
			imgsz=640,
			batch=batch,
			name=modelName
		)
		print(f"✅ Training completed: {modelName}")
		return results

	def trainPipeline(self):
		"""Run full pipeline: base training + fine-tuning."""
		print("🚀 Starting COCO training...")
		self.trainModel("data/coco.yaml", epochs=20, batch=16, modelName="yolov8CocoBase")

		print("\n🔧 Fine-tuning on KITTI dataset...")
		self.trainModel("data/kitti.yaml", epochs=10, batch=8, modelName="yolov8KittiFinetune")

if __name__ == "__main__":
	trainer = YOLOTrainer()
	trainer.trainPipeline()
