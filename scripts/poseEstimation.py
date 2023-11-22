from ultralytics import YOLO
from scripts.sort import *
import cv2
import cvzone
import pandas as pd


class PoseEstimator:
    def __init__(self, tracker_max_age=20, tracker_min_hits=3, tracker_iou_threshold=0.3):
        self.model = YOLO('models/yolov8n-pose.pt')

        # coco dataset class names
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                           "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                           "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"
                           ]

        self.tracker = Sort(max_age=tracker_max_age, min_hits=tracker_min_hits, iou_threshold=tracker_iou_threshold)

    def findPose(self, inFrame, draw_detection_bbox=True, draw_detection_tag=True, draw_pose_landmark=True):
        df = pd.DataFrame(
            columns=['Id', 'Bbox', 'lm1', 'lm2', 'lm3', 'lm4', 'lm5', 'lm6', 'lm7', 'lm8', 'lm9', 'lm10', 'lm11',
                     'lm12', 'lm13', 'lm14', 'lm15', 'lm16'])
        lm_list = []
        results = self.model(source=inFrame)
        detections = np.empty((0, 5))
        for result in results:
            for idx, xyxy in enumerate(result.boxes.xyxy.numpy()):
                lm_dict = {}
                x = int(xyxy[0])
                y = int(xyxy[1])
                x1 = int(xyxy[2])
                y1 = int(xyxy[3])
                currentClass = self.classNames[int(result.boxes.cls[idx])]
                conf = result.boxes.conf[idx].numpy()
                if currentClass == 'person' and conf > 0.3:
                    lm_dict.update({'bbox': [x, y, x1, y1]})
                    keypoint = result.keypoints.xy[idx].numpy()
                    for idx, kp in enumerate(keypoint):
                        xpt = int(kp[0])
                        ypt = int(kp[1])
                        lm_dict.update({'lm{}'.format(idx + 1): [xpt, ypt]})
                        if draw_pose_landmark:
                            cv2.circle(inFrame, (xpt, ypt), 2, (0, 0, 255), 2)
                    currentArray = np.array([x, y, x1, y1, conf])
                    detections = np.vstack([detections, currentArray])
                    lm_list.append(lm_dict)

        resultsTracker = self.tracker.update(detections)
        for result in resultsTracker:
            x, y, x1, y1, Id = result
            x, y, x1, y1, Id = int(x), int(y), int(x1), int(y1), int(Id)
            for lm in lm_list:
                if x - 5 < lm['bbox'][0] < x + 5 and \
                        y - 5 < lm['bbox'][1] < y + 5 and \
                        x1 - 5 < lm['bbox'][2] < x1 + 5 and \
                        y1 - 5 < lm['bbox'][3] < y1 + 5:
                    df.loc[len(df)] = {"Id": Id,
                                       "Bbox": lm['bbox'],
                                       "lm1": lm['lm1'],
                                       "lm2": lm['lm2'],
                                       "lm3": lm['lm3'],
                                       "lm4": lm['lm4'],
                                       "lm5": lm['lm5'],
                                       "lm6": lm['lm6'],
                                       "lm7": lm['lm7'],
                                       "lm8": lm['lm8'],
                                       "lm9": lm['lm9'],
                                       "lm10": lm['lm10'],
                                       "lm11": lm['lm11'],
                                       "lm12": lm['lm12'],
                                       "lm13": lm['lm13'],
                                       "lm14": lm['lm14'],
                                       "lm15": lm['lm15'],
                                       "lm16": lm['lm16']}
            w, h = x1 - x, y1 - y
            if draw_detection_bbox:
                cvzone.cornerRect(inFrame, (x, y, w, h), l=9, rt=2, colorR=(255, 0, 0))
            if draw_detection_tag:
                cvzone.putTextRect(inFrame, str(Id), (max(0, x), max(35, y)),
                                   scale=2, thickness=3, offset=10)
        return df
