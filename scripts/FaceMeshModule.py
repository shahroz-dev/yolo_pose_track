import time
import cv2
import mediapipe as mp
import os


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, refLm=False,
                 minDtCf=0.5, minTrCf=0.5):
        self.results = None
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refLm = refLm
        self.minDtCf = minDtCf
        self.minTrCf = minTrCf
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.refLm, self.minDtCf, self.minTrCf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceId, faceLms in enumerate(self.results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for lmId, lm in enumerate(faceLms.landmark):
                    # print(id, lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print(faceId, lmId, x, y)
                    face.append([x, y])
                    # cv2.putText(img, str(lmId), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                faces.append(face)
        return img, faces


def main():
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    dst_path = "Saved_Results/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    videoWriter = cv2.VideoWriter(dst_path + 'faceMesh.avi', fourcc, 30.0, (640, 480))
    frameWidth = 640
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(refLm=True)
    while cap.isOpened():
        success, img = cap.read()
        if success:

            img, faces = detector.findFaceMesh(img)
            if len(faces) != 0:
                # print(len(faces), faces[0])
                for id, lm in enumerate(faces[0]):
                    print(id, lm[0], lm[1])

            r = frameWidth / img.shape[1]  # width height ratio
            dim = (frameWidth, int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, 'FPS: {}'.format(str(int(fps))), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.imshow("Video", img)
            videoWriter.write(img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
