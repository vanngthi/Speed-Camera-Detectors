import cv2
import numpy as np
from datetime import datetime

import easyocr
from ultralytics import YOLO

from sort import *
from util import *
from object_detection import ObjectDetection

# Khai báo phát hiện đối tượng
od = ObjectDetection()
class_names = od.load_class_names()

# Khởi tạo Ocr
reader = easyocr.Reader(["en"])

# Khởi tạo tracker dùng thuật toán Sort
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Tạo từ điển
vehicles_entering = {} # Lưu trữ đối tượng
vehicles_speed = {} # Lưu trữ thời gian của đối tượng

# Đọc video
cap = cv2.VideoCapture("../../data/video/plate.mp4")
license_plate_detector = YOLO('../model/yolov8/license_plate_detector.pt')
fps = cap.get(cv2.CAP_PROP_FPS) # Số lượng frame trong 1s

# Tạo video đầu ra
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc thử *'X264'
out = cv2.VideoWriter('../../output/speed_cam/speedcam_plate.mp4', fourcc, fps, (width, height))

# Đường dẫn đến file txt
output_file_path = "../../output/speed_cam/result.txt"

# Tạo đường line
distance = 5
line1 = [(446, 700), (1450, 700)]
line2 = [(446, 500), (1400, 500)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = od.detect(frame)
    boxes = result[0].boxes

    detections = np.empty((0, 5))
    for box in boxes:
        (x1, y1, x2, y2) = np.array(box.xyxy[0], dtype=int)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        conf = np.array(boxes.conf[0], dtype=float)
        cls = np.array(boxes.cls[0], dtype=int)
        currentClass = class_names[cls]

        if currentClass in ["car", "truck", "bus"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    resultsTracker = tracker.update(detections)

    # Ước lượng vận tốc
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        cx, cy  = x1 + w//2, y1 + h//2
        # cv2.putText(frame, str(id), (cx, cy), 0, 0.5, (255, 255, 255), 2)
        if id not in vehicles_entering and id not in vehicles_speed:
            if cy <=  line1[0][1] + 7 and cy >= line1[0][1] - 7 and cx <= line2[1][0] and cy > line2[0][1]:
                vehicles_entering[id] = 0

        if id in vehicles_entering:
            if cy >= line2[1][1]:
                vehicles_entering[id] = vehicles_entering[id] + 1
            else:
                # Tính toán vận tốc
                elapsed_time = vehicles_entering[id]*1/fps
                a_speed_ms = distance/elapsed_time
                a_speed_kh = a_speed_ms* 3.6

                # Khoanh vùng các xe vi phạm vận tốc
                if a_speed_kh >= 25:
                    # Lấy thời gian hiện tại
                    current_time = datetime.now()
                    image_car = frame[y1:y2, x1:x2, :]
                    # cv2.imshow("image car", image_car)
                    car_path = f"../../output/speed_cam/{id}.png"
                    cv2.imwrite(car_path, image_car)
                    plx1, ply1 , plx2, ply2, _,_ = license_plate_detector(image_car)[0].boxes.data.tolist()[0]
                    # print(plx1, plx2, ply1, ply2)
                    crop_plate = image_car[int(ply1):int(ply2), int(plx1):int(plx2),:]
                    # print(crop_plate.shape)
                    if crop_plate.shape[0] > 0 and crop_plate.shape[1] > 0:
                        res_plate = reader.readtext(crop_plate)
                        res = res_plate[0][1].upper().replace(' ', '')

                        # Kiểm tra định dạng biển số xe
                        if license_format(res):
                            plate = format_license(res)
                            vehicles_speed[id] = a_speed_kh, plate, car_path, current_time.strftime("%Y-%m-%d_%H:%M:%S")
                        else:
                            vehicles_speed[id] = a_speed_kh, res, car_path, current_time.strftime("%Y-%m-%d_%H:%M:%S")
                    else:
                        vehicles_speed[id] = a_speed_kh, "None", car_path, current_time.strftime("%Y-%m-%d_%H:%M:%S")
                del vehicles_entering[id]

        if id in vehicles_speed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 170, 66), 2)
            cv2.rectangle(frame, (x1, y1), (x1+250, y1-20), (245, 170, 66), -1)
            cv2.putText(frame, str(round(vehicles_speed[id][0], 2)) + "km/h", (x1, y1-5), 0, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, str(vehicles_speed[id][1]), (x1 + 100, y1-5), 0, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (245, 170, 66), -1)

    cv2.line(frame, line1[0], line1[1], (15, 220, 10), 2)
    cv2.line(frame, line2[0], line2[1], (15, 220, 10), 2)
    out.write(frame)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

with open(output_file_path, 'w') as file:
    file.writelines(f"{key}: {value}\n" for key, value in vehicles_speed.items())

cap.release()
out.release()
cv2.destroyAllWindows()