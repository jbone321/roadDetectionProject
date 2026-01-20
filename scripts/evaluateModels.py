from ultralytics import YOLO
import matplotlib.pyplot as plt

class ModelEvaluator:
	"""Evaluates YOLOv8 models for object detection performance."""
	def __init__(self, modelPath, datasetYaml):
		self.modelPath = modelPath
		self.datasetYaml = datasetYaml
		self.model = YOLO(modelPath)

	def evaluateModel(self):
		"""Run model validation and print key metrics."""
		metrics = self.model.val(data=self.datasetYaml)
		print("Evaluation Summary:")
		print(f"mAP50: {metrics.box.map50:.4f}") # Mean Average Precision at IoU=0.5
		print(f"mAP50-95: {metrics.box.map:.4f}") # Mean Average Precision at IoU=0.5:0.95
		print(f"Inference Speed: {metrics.speed["inference"]:.2f} ms/img") # Inference speed
		return metrics

	def visualizePredictions(self, samplePath):
		"""Display predictions for one sample image."""
		results = self.model(samplePath, show=False)
		img = results[0].plot()
		plt.imshow(img)
		plt.axis("off")
		plt.title("Predicted Detections")
		plt.show()

if __name__ == "__main__":
	evaluator = ModelEvaluator(
		modelPath="../runs/detect/yolov8_kitti_finetune/weights/best.pt",
		datasetYaml="../data/kitti.yaml"
	)
	evaluator.evaluateModel()
	evaluator.visualizePredictions("../data/kitti/data_object_image_2/testing/image_2/000050.png")
