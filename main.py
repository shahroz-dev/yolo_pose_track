from scripts.sort import *
import cv2
import scripts.poseEstimation as poseEstimator
import pandas as pd


def main():
    cap = cv2.VideoCapture('videos/ch09_20231111000356.mp4')
    targetHeight = 480

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    dst_path = "Saved_Results/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    videoWriter = cv2.VideoWriter(dst_path + 'YOLO_based_pose_detection.avi', fourcc, 30.0,
                                  (int(targetHeight * 1.7777777777777777), targetHeight))

    poseEst = poseEstimator.PoseEstimator()

    df = pd.DataFrame()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            height, width, _ = frame.shape
            r = width / height
            frame = cv2.resize(frame, (int(targetHeight * r), targetHeight))
            df_temp = poseEst.findPose(inFrame=frame)
            df = pd.concat([df, df_temp], ignore_index=True)
            cv2.imshow('result', frame)
            videoWriter.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    file_name = 'pose_landmarks.xlsx'
    df.to_excel(file_name)


if __name__ == "__main__":
    main()
