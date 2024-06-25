import cv2
import numpy as np
from object_detection import ObjectDetection
import math

# Khởi tạo Object Detection sử dụng yolov8
od = ObjectDetection()
class_names = od.load_class_names()
# Nhập video đầu vào
cap = cv2.VideoCapture("../../../data/video/traffic.mp4")

# Xác định thuộc tính của video đầu vào
# width = int(cap.get(3))
# height = int(cap.get(4))
# fps = cap.get(5)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # hoặc thử *'X264'
# out = cv2.VideoWriter('../../output/tracking/base_traffic.mp4', fourcc, fps, (width, height))

# Khởi tạo biến
count = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Mảng lưu trữ tâm của đối tượng trong frame
    center_points_cur_frame = []

    # Detect đối tượng trong frame
    result = od.detect(frame)
    class_ids = np.array(result[0].boxes.cls, dtype=int)
    scores = np.array(result[0].boxes.conf, dtype=float)
    boxes = np.array(result[0].boxes.xyxy, dtype=int)
    currentClass = class_names[class_ids[0]]

    for box in boxes:
        (x1, y1, x2, y2) = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy  = x1 + w//2, y1 + h//2
        center_points_cur_frame.append((cx, cy)) # Tâm của object trong frame hiện tại

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Mới bắt đầu, so sánh 2 khung hình đầu tiên
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 10:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update vị trí mới cho object 
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Bỏ bớt các ID
            if not object_exists:
                tracking_objects.pop(object_id)

        # Thêm ID mới
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        # cv2.putText(frame, currentClass, (pt[0], pt[1] + 20), 0, 1, (0, 0, 255), 2)

    # out.write(frame)
    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
