from scripts.sort import *
import cv2
import scripts.poseEstimation as poseEstimator
import csv
import os

class Annotator(object):
    def __init__(self, label):
        self.label = label
        self.poseEst = poseEstimator.PoseEstimator()

    def get_annotated_frame(self, img_path, csv_writer):
        img = cv2.imread(img_path)
        tracked_lm_list = self.poseEst.findPose(inFrame=img)
        for lm in tracked_lm_list:
            lm.update({"label": self.label})
            csv_writer.writerow(lm)

def get_all_images(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    base_output_folder = "path/to/your/extracted/frames"  # Replace with the path to your extracted frames
    csv_file = "annotated_data.csv"

    image_files = get_all_images(base_output_folder)
    annotator = Annotator('sober')  # Change to 'drunk' as needed

    with open(csv_file, 'w', newline='') as csvfile:
        headers = ["Track_Id", "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2",
                   "lm1_x", "lm1_y", "lm2_x", "lm2_y", "lm3_x", "lm3_y", "lm4_x", "lm4_y",
                   "lm5_x", "lm5_y", "lm6_x", "lm6_y", "lm7_x", "lm7_y", "lm8_x", "lm8_y",
                   "lm9_x", "lm9_y", "lm10_x", "lm10_y", "lm11_x", "lm11_y", "lm12_x", "lm12_y",
                   "lm13_x", "lm13_y", "lm14_x", "lm14_y", "lm15_x", "lm15_y", "lm16_x", "lm16_y", "label"]
        csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
        csv_writer.writeheader()

        for img_path in image_files:
            annotator.get_annotated_frame(img_path, csv_writer)

if __name__ == "__main__":
    main()
