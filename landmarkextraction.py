import cv2
import scripts.poseEstimation as poseEstimator
import csv
import os
import re

class Annotator(object):
    def __init__(self, frame_rate, state):
        self.frame_rate = frame_rate
        self.state = state
        self.poseEst = poseEstimator.PoseEstimator()

    def get_annotated_frame(self, img_path, csv_writer):
        img = cv2.imread(img_path)
        df = self.poseEst.findPose(inFrame=img)

        # Extracting frame_number from the filename
        filename = os.path.basename(img_path)
        match = re.match(r"frame_(\d+).jpg", filename)  # Assuming filenames like 'frame_0001.jpg'
        if match:
            frame_number = int(match.group(1))
            timestamp = frame_number / self.frame_rate

            for index, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict.update({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "label": self.state
                })
                csv_writer.writerow(row_dict)

def get_subdirectories(folder_path):
    return [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

def get_all_images(folder_path):
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        files.sort()  # Sort files to maintain the sequence
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    base_folder = "Drunk_Frames"  # Replace with your base folder
    csv_file = "annotated_data.csv"
    frame_rate = 30  # Replace with your videos' frame rate
    state = "drunk"  # Assuming the state is 'drunk'

    subdirectories = get_subdirectories(base_folder)


    with open(csv_file, 'w', newline='') as csvfile:
        headers = ["Track_Id", "Bbox_x1", "Bbox_y1", "Bbox_x2", "Bbox_y2",
                   "lm1_x", "lm1_y", "lm2_x", "lm2_y", "lm3_x", "lm3_y", "lm4_x", "lm4_y",
                   "lm5_x", "lm5_y", "lm6_x", "lm6_y", "lm7_x", "lm7_y", "lm8_x", "lm8_y",
                   "lm9_x", "lm9_y", "lm10_x", "lm10_y", "lm11_x", "lm11_y", "lm12_x", "lm12_y",
                   "lm13_x", "lm13_y", "lm14_x", "lm14_y", "lm15_x", "lm15_y", "lm16_x", "lm16_y", "label",
                  "frame_number", "timestamp", "label"]
        csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
        csv_writer.writeheader()

        for subdir in subdirectories:
            image_files = get_all_images(subdir)
            annotator = Annotator(frame_rate, state)

            for img_path in image_files:
                annotator.get_annotated_frame(img_path, csv_writer)

if __name__ == "__main__":
    main()
