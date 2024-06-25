import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path="../../model/yolov8/yolov8n.pt"):
        print("Loading Object Detection")

        self.model = YOLO(model_path)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

    def load_class_names(self, classes_path="../../model/yolov8/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model(frame)