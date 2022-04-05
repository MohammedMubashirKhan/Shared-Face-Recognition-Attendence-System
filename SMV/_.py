import pickle
import time
import face_recognition
import cv2
import Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
Localmp = medoapipe_face_mesh.FaceMesh()
# import SMV.Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
# localMP = medoapipe_face_mesh.FaceMesh()
# import SMV.Face_Mesh.mediapipe_face_detection as mediapipe_face_detection
# mp_faceDetection = mediapipe_face_detection.FaceDetection()
cap =cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

pastTime = time.time()
while True:
    succ, frame = cap.read()
    if not succ:
        print("video Ended")
        break
    frame = cv2.resize(frame, (0,0), fx=20, fy=20)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_meshs_custom_landmarks = Localmp.faceMeshDetection(frameRGB)
    face_locations = Localmp.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)
    if face_locations:
        print("face fnoubd is:", len(face_locations))
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left,top), (right, bottom),
                    color=(0,0,255),
                    thickness=5)
    frame = cv2.resize(frame, (1080,720))
    currentTime = time.time()
    fps = 1/ (currentTime-pastTime)
    pastTime = currentTime
    cv2.putText(frame, str(fps), (20,50), cv2.FONT_HERSHEY_PLAIN, color=(0,0,255), thickness=2, fontScale=2)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
## processing face mesh and storing all point in a list in pixel form
## for processing face mesh the function required image in RGB format
# face_meshs_custom_landmarks = face_recognition.face_locations(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), model="dnn")
# print(len(face_meshs_custom_landmarks))
# print(face_meshs_custom_landmarks)
# # localMP.drawFaceMeshPoints(frame, face_meshs_custom_landmarks=face_meshs_custom_landmarks)
# face_locations = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)

cv2.waitKey(0)