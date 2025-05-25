import torch

class FoodDetector:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect(self, image_path):
        results = self.model(image_path)
        return results.pandas().xyxy[0]  # Returns bounding boxes and labels

