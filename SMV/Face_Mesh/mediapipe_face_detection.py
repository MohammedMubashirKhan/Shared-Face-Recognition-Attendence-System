import cv2
import mediapipe as mp

class FaceDetection:

    def __init__(self) -> None:
        self.mp_faceDetection = mp.solutions.mediapipe.python.solutions.face_detection.FaceDetection()
        self.mp_draw = mp.solutions.drawing_utils

    def faceDetection(self, image):
        results = self.mp_faceDetection.process(image=image)
        if not results.detections:
            print("No face is Detected")
            return
        for detection in results.detections:
            print("Face is Detected")
            self.mp_draw.draw_detection(image=image, detection=detection)
            # print(detection)
