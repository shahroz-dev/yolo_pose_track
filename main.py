import cvzone
from scripts.sort import *
import cv2
import scripts.poseEstimation as poseEstimator
import scripts.FaceMeshModule as fm
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/gait_binary_model.tflite")
from collections import deque
import csv


class Camera(object):
    frame_id = 0
    frame_width = 480
    frame_height = 360

    def __init__(self):
        self.video = cv2.VideoCapture('videos/ch09_20231111000000.mp4')
        # self.video = cv2.VideoCapture(0)
        self.video.set(3, Camera.frame_width)
        self.video.set(4, Camera.frame_height)
        self.poseEst = poseEstimator.PoseEstimator()
        self.faceMeshEst = fm.FaceMeshDetector(refLm=True)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_annotated_frame(self):
        while self.video.isOpened():
            success, frame = self.video.read()
            if success:
                # Person Detector and Tracker with Pose Landmark
                tracked_lm_list = self.poseEst.findPose(inFrame=frame)
                for lm in range(len(tracked_lm_list)):
                    landmarks_np = np.array([[tracked_lm_list[lm]["lm"+str(i)+"_x"], tracked_lm_list[lm]["lm"+str(i)+"_y"]]
                                             for i in range(1, 17)]).reshape(1, -1)
                    prediction = model.predict(landmarks_np)[0]
                    state = "Drunk" if prediction[0] > 0.5 else "Sober"
                    cvzone.putTextRect(frame, state,
                                       (tracked_lm_list[lm]["Bbox_x2"], tracked_lm_list[lm]["Bbox_y1"]), scale=2)
                    self.write_landmarks_to_csv(tracked_lm_list[lm], 'rolling_data.csv')

                # Depth Estimator
                img, faces = self.faceMeshEst.findFaceMesh(frame, draw=False)
                if faces:
                    for face in range(len(faces)):
                        pointLeft = faces[face][145]
                        pointRight = faces[face][374]
                        w = ((pointRight[0] - pointLeft[0]) ** 2 + (pointRight[1] - pointLeft[1]) ** 2) ** (
                                1 / 2)  # pixel distance between pupils
                        W = 6.3  # average value in cm between pupils
                        f = 1000  # average focal length value of a webcam
                        d = (W * f) / w
                        cvzone.putTextRect(frame, 'Depth: {}cm'.format(int(d)),
                                           (faces[face][10][0] - 100, faces[face][10][1] - 50), scale=2)

                cv2.imshow('result', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                break

            # ret, frame = self.video.read()
            #
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # results = pose.process(frame_rgb)
            #
            # confidence_values = (None, None)  # Initialize both pose and pupil confidence to None
            #
            # Camera.frame_id += 1
            #
            # if results.pose_landmarks:
            #     landmarks = results.pose_landmarks.landmark
            #     self.write_landmarks_to_csv(landmarks, 'rolling_data.csv')
            #     landmarks = [landmark for landmark in results.pose_landmarks.landmark]
            #     frame_width = frame.shape[1]
            #     frame_height = frame.shape[0]
            #     xs = [landmark.x * frame_width for landmark in landmarks]
            #     ys = [landmark.y * frame_height for landmark in landmarks]
            #     x_min, x_max = min(xs), max(xs)
            #     y_min, y_max = min(ys), max(ys)
            #     cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            #     landmarks_np = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).reshape(1, -1)
            #     prediction = model.predict(landmarks_np)[0]
            #     state = "Drunk" if prediction[0] > 0.5 else "Sober"
            #     confidence = max(prediction[0], prediction[1]) * 100
            #     text_position = (int(x_min), int(y_min) - 10)
            #     cv2.putText(frame, f"Prediction: {state} ({confidence:.2f}%)", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
            #                 (0, 0, 255), 2)
            #     confidence_values = (confidence / 100, None)  # Set pose confidence
            # else:
            #     frame, pupil_confidence = self.detect_pupil(frame)
            #     confidence_values = (confidence_values[0], pupil_confidence)
            #
            # ret, jpeg = cv2.imencode('.jpg', frame)
            # return jpeg.tobytes(), confidence_values

    @staticmethod
    def write_landmarks_to_csv(landmarks, filename):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            headers = ["Track_Id", "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2",
                       "lm1_x", "lm1_y", "lm2_x", "lm2_y", "lm3_x", "lm3_y", "lm4_x", "lm4_y",
                       "lm5_x", "lm5_y", "lm6_x", "lm6_y", "lm7_x", "lm7_y", "lm8_x", "lm8_y",
                       "lm9_x", "lm9_y", "lm10_x", "lm10_y", "lm11_x", "lm11_y", "lm12_x", "lm12_y",
                       "lm13_x", "lm13_y", "lm14_x", "lm14_y", "lm15_x", "lm15_y", "lm16_x", "lm16_y"]
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # Write the header if the file is new
            writer.writerow(landmarks)


def main():
    camera = Camera()
    camera.get_annotated_frame()


if __name__ == "__main__":
    main()
