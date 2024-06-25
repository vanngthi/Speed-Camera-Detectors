from ultralytics import YOLO
import cv2

import util
from sort import *
from util import get_car, read_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO("../model/yolov8/yolov8n.pt")
license_plate_detector = YOLO('D:/study/CV/CK/speechcam_detection/code/plate/license_plate_detector.pt')

cap = cv2.VideoCapture('D:/study/CV/CK/speechcam_detection/data/video/plate1.mp4')

vehicles = [2, 3, 5, 7]

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # theo dõi đối tượng
        track_ids = mot_tracker.update(np.asarray(detections_))

        # dectec biển số
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # xét biển số của xe nào
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # ảnh biển số 
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # đọc ký tự
                license_plate_text, license_plate_text_score = read_plate(license_plate_crop)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test2.csv')