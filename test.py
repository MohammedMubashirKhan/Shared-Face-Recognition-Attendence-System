# from sklearn import svm
import pickle
from time import time
import cv2
import numpy as np
import Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
import face_recognition as fr
localMP = medoapipe_face_mesh.FaceMesh()
from Face_Mesh.mediapipe_face_detection import FaceDetection

fd = FaceDetection()
name = input("What is your name\n")

cap = cv2.VideoCapture(0)
frame_width = cap.get(3) / 4
frame_width = int (frame_width)
frame_height = cap.get(4) / 4
frame_height = int (frame_height)
pastTime = 0
count = 0
process_this_frame = True
only_one = True
face_encoding_average = np.array(128) * 0
while(count < 100):
    success, frame = cap.read()
    if success != True or cv2.waitKey(1) == ord('q'):
        print("Video has ended")
        break
    count +=1
    # print(frame.shape)
    # frame = cv2.flip(frame,1)
    small_frame = cv2.resize(frame, dsize=(frame_width, frame_height), dst=-1,)
    small_frameRGB = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    cv2.imshow("small_frame", small_frame)


    ## processing face mesh and storing all point in a list in pixel form
    ## for processing face mesh the function required image in RGB format
    face_meshs_custom_landmarks = localMP.faceMeshDetection(small_frameRGB, enlargeby=4)
    # localMP.drawFaceMeshPoints(frame, face_meshs_custom_landmarks=face_meshs_custom_landmarks)
    faceLocation = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)
    # print("faceLocation : ",faceLocation)
    # face_encodings = []

    try:
        for (xmin,ymin,xmax,ymax) in faceLocation:
            # xmin = xmin * 4 
            # ymin = ymin * 4 
            # xmax = xmax * 4 
            # ymax = ymax * 4 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),color=(0,0,255),thickness=2)
            # cv2.imshow("Croped", frame[ymin:ymax,xmin:xmax,0:3])
    #         cv2.imshow("CropedRGB", frameRGB[ymin:ymax,xmin:xmax,0:3])
    #         face_encoding = fr.face_encodings(face_image=frameRGB)
    #         face_encodings.append(face_encoding[0])
    #         print(face_encoding[0]) 
    #         print()
    #         count += 1
    #         if count == 2:
    #             exit()
    except Exception as e:
        print(e)
    
    if only_one and faceLocation:
        for (xmin,ymin,xmax,ymax) in faceLocation:
            face_Encoding_only_one = fr.face_encodings(face_image=frame, known_face_locations=[(ymin)])[0]
            only_one =False

    if process_this_frame and faceLocation:
        face_encodings = fr.face_encodings(frame, faceLocation)
        if face_encodings:
            test = face_encodings[0]
            face_encoding_average = (face_encoding_average+test)/2
    # process_this_frame = not process_this_frame

    # faces = fr.face_locations(small_frameRGB)
    # print("face_recognition :",faces)
    # for (top, right, bottom, left) in faces:
    # #     print("Inside forr loop")
    #     top *=4
    #     right *=4
    #     bottom *=4
    #     left *=4
    #     cv2.rectangle(frame, (left,top),(right,bottom),
    #                 color=(0,255,0), thickness=2)
    #     cv2.imshow("Croped", frameRGB[top:bottom,left:right,0:3])
    #     faceEncoding = fr.face_encodings(frameRGB[top:bottom,left:right,0:3])


















    currentTime = time()
    fps = 1/(currentTime - pastTime)
    fps = int(fps)

    pastTime = currentTime
    cv2.putText(frame, str(fps)+" FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,120,255), thickness=2)
    cv2.imshow("frame", frame)
cv2.destroyAllWindows()



data = {
    "Mubashir Khan":face_Encoding_only_one,
    "Mubashir averge":face_encoding_average
}
all_face_encoding = {}
try:
    with open("face_Encoding.dat", "rb") as f:
        all_face_encoding = pickle.load(f)
        all_face_encoding[name] = face_Encoding_only_one
except:
    all_face_encoding[name] = face_Encoding_only_one
    

with open("face_Encoding.dat", "wb") as f:
    pickle.dump(all_face_encoding, f)


face_name = list(all_face_encoding.keys())
face_encodings = np.array(list(all_face_encoding.values()))


print(face_name)
print(face_encodings)



while True:
    success, frame = cap.read()
    if success != True or cv2.waitKey(1) == ord('q'):
        print("Video has ended")
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # fd.faceDetection(image=frameRGB)

    face_meshs_custom_landmarks = localMP.faceMeshDetection(image=frameRGB)
    faceLocation = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)

    if faceLocation:
        unKnow_face = fr.face_encodings(face_image=frameRGB, known_face_locations=faceLocation)
        # for i in range(len(face_name)):
        # print(unKnow_face)
        # print(len(unKnow_face))
        result = fr.compare_faces(face_encodings, unKnow_face, tolerance=0.45)
        print(result)
        for i in range(len(result)):
            if result[i] == True:
                break

    else:
        pass
        # result =False

    
    # frameRGB = cv2.flip(frameRGB,1)
    currentTime = time()
    fps = int(1/(currentTime - pastTime))
    pastTime = currentTime

    if result[i]:
        # print(result)
        for (x1,y1,x2,y2) in faceLocation:
            cv2.rectangle(frame, (x1, y1), (x2, y2),color=(0,255,0),thickness=1)
            cv2.putText(frame, face_name[i], (x1,y1+20), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,255), thickness=1)
    else:
        for (x1,y1,x2,y2) in faceLocation:
            cv2.rectangle(frame, (x1, y1), (x2, y2),color=(0,0,255),thickness=2)
            cv2.putText(frame, "UnKnow Face", (x1,y1+20), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,255), thickness=1)


    cv2.putText(frame, str(fps)+" FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,120,255), thickness=2)
    cv2.imshow("face", frame)





cv2.destroyAllWindows()
cap.release()