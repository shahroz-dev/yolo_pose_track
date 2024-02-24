from ultralytics import YOLO
from scripts.sort import *
import cv2
import cvzone


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
        tracked_lm_list = []
        for result in resultsTracker:
            x, y, x1, y1, Id = result
            x, y, x1, y1, Id = int(x), int(y), int(x1), int(y1), int(Id)
            for lm in lm_list:
                if x - 5 < lm['bbox'][0] < x + 5 and \
                        y - 5 < lm['bbox'][1] < y + 5 and \
                        x1 - 5 < lm['bbox'][2] < x1 + 5 and \
                        y1 - 5 < lm['bbox'][3] < y1 + 5:
                    tracked_lm_list.append({"Track_Id": Id,
                                            "Bbox_x1": lm['bbox'][0],
                                            "Bbox_y1": lm['bbox'][1],
                                            "Bbox_x2": lm['bbox'][2],
                                            "Bbox_y2": lm['bbox'][3],
                                            "lm1_x": lm['lm1'][0],
                                            "lm1_y": lm['lm1'][1],
                                            "lm2_x": lm['lm2'][0],
                                            "lm2_y": lm['lm2'][1],
                                            "lm3_x": lm['lm3'][0],
                                            "lm3_y": lm['lm3'][1],
                                            "lm4_x": lm['lm4'][0],
                                            "lm4_y": lm['lm4'][1],
                                            "lm5_x": lm['lm5'][0],
                                            "lm5_y": lm['lm5'][1],
                                            "lm6_x": lm['lm6'][0],
                                            "lm6_y": lm['lm6'][1],
                                            "lm7_x": lm['lm7'][0],
                                            "lm7_y": lm['lm7'][1],
                                            "lm8_x": lm['lm8'][0],
                                            "lm8_y": lm['lm8'][1],
                                            "lm9_x": lm['lm9'][0],
                                            "lm9_y": lm['lm9'][1],
                                            "lm10_x": lm['lm10'][0],
                                            "lm10_y": lm['lm10'][1],
                                            "lm11_x": lm['lm11'][0],
                                            "lm11_y": lm['lm11'][1],
                                            "lm12_x": lm['lm12'][0],
                                            "lm12_y": lm['lm12'][1],
                                            "lm13_x": lm['lm13'][0],
                                            "lm13_y": lm['lm13'][1],
                                            "lm14_x": lm['lm14'][0],
                                            "lm14_y": lm['lm14'][1],
                                            "lm15_x": lm['lm15'][0],
                                            "lm15_y": lm['lm15'][1],
                                            "lm16_x": lm['lm16'][0],
                                            "lm16_y": lm['lm16'][1]
                                            })
            w, h = x1 - x, y1 - y
            if draw_detection_bbox:
                cvzone.cornerRect(inFrame, (x, y, w, h), l=9, rt=2, colorR=(255, 0, 0))
            if draw_detection_tag:
                cvzone.putTextRect(inFrame, str(Id), (max(0, x), max(35, y)),
                                   scale=2, thickness=3, offset=10)
        return tracked_lm_list
