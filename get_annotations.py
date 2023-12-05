from scripts.sort import *
import cv2
import scripts.poseEstimation as poseEstimator
import tensorflow as tf
model = tf.keras.models.load_model("models/gait_model.h5")
import csv


class Annotator(object):
    frame_id = 0
    frame_width = 480
    frame_height = 360

    def __init__(self, img_path, label):
        self.img = cv2.imread(img_path)
        self.label = label
        self.poseEst = poseEstimator.PoseEstimator()

    def get_annotated_frame(self):
        tracked_lm_list = self.poseEst.findPose(inFrame=self.img)
        for lm in range(len(tracked_lm_list)):
            self.write_landmarks_to_csv(tracked_lm_list[lm], self.label, 'annotated_data.csv')

    @staticmethod
    def write_landmarks_to_csv(landmarks, label, filename):
        landmarks.update({"label": label})
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as csvfile:
            headers = ["Track_Id", "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2",
                       "lm1_x", "lm1_y", "lm2_x", "lm2_y", "lm3_x", "lm3_y", "lm4_x", "lm4_y",
                       "lm5_x", "lm5_y", "lm6_x", "lm6_y", "lm7_x", "lm7_y", "lm8_x", "lm8_y",
                       "lm9_x", "lm9_y", "lm10_x", "lm10_y", "lm11_x", "lm11_y", "lm12_x", "lm12_y",
                       "lm13_x", "lm13_y", "lm14_x", "lm14_y", "lm15_x", "lm15_y", "lm16_x", "lm16_y", "label"]
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # Write the header if the file is new
            writer.writerow(landmarks)


def main():
    annotator = Annotator('Data/drunk/Drunk1_frame_0045.jpg', 'drunk')
    annotator.get_annotated_frame()


if __name__ == "__main__":
    main()
