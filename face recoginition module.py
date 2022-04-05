import sys
sys.path.append('C:\\Users\\User\\Desktop\\Projests Practicing for self\\Image Processing\\Face Recognition Attendence System\\Face Mesh')

import cv2
import time
from matplotlib.pyplot import draw
import numpy as np
import mediapipe as mp
import medoapipe_face_mesh


# ALL IS TRANSFER TO MEDIAPIPE_FACE_MESH FILE IN A FACEMESH CLASS

# # It takes a frame of an image and return list of face meshs
# # This function return a list of all face mesh
# def faceMeshDetection (image):
#     face_mesh_result = face_mesh.process(image)

#     face_meshs_custom_landmarks=[]

#     if face_mesh_result.multi_face_landmarks:
#         for face_landmarks in face_mesh_result.multi_face_landmarks:
            
#             face_mesh_custom_landmark = []
#             for landmark in face_landmarks.landmark:
#                 x = landmark.x * frame_width
#                 x = int (x)
#                 y = landmark.y * frame_height
#                 y = int(y)
#                 z = landmark.z 
#                 face_mesh_custom_landmark.append((x, y, z))
#             face_meshs_custom_landmarks.append(face_mesh_custom_landmark)
#             # print((face_meshs_custom_landmarks[0][0][0]))
#     return face_meshs_custom_landmarks



def handDetection(image):
    hand_result = hand_detect.process(image)
    hand_landmark = []
    if hand_result.multi_hand_landmarks:
        for hand in hand_result.multi_hand_landmarks:
            for landmark in hand.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                z = landmark.z
                hand_landmark.append((x, y, z))
                # print(x)
                # print(y)
                # print(z)
    return hand_landmark
            



mp_hand = mp.solutions.mediapipe.python.solutions.hands
hand_detect = mp_hand.Hands(max_num_hands=6)



mp_draw = mp.solutions.mediapipe.python.solutions.drawing_utils

localMP = medoapipe_face_mesh.FaceMesh()


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))



print(frame_width)
print(frame_height)

# past time for calculating FPS
pt = 0
point = 0
while True:
    if point == 468:
        point = 0

    cp = time.time()
    fps =  int(1 / (cp - pt))
    pt = cp
    
    success, frame = cap.read()
    if success != True:
        print("Video Ended")
        break
    # frame = cv2.resize(frame, (0,0), dst=-1, fx=2, fy=2)
    frameRGB = frame[:,:,::-1]
    frameCopy = frame.copy()

    # Getting all faces mesh in x,y coordinate in a list an z cordinate also but it's normalize value
    face_meshs_custom_landmarks = localMP.faceMeshDetection(frameRGB, frame_width=frame_width, frame_height=frame_height)

    hand_landmarks =  handDetection(frameRGB)


    facesLocations = []
    minX=minY=0
    maxX=maxY=0

    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    for hand_landmark in hand_landmarks:
        x , y, z = hand_landmark
        # cv2.circle(frame, (x, y), 3, color=(0,255,0), thickness=-1)
        
        cv2.circle(canvas, (x, y), 3, color=(0,255,0), thickness=-1)

    for face in face_meshs_custom_landmarks:
        faceLocation = []
        
        cv2.circle(frame, face[point][0:2], 3, color=(0,255,0), thickness=-1)
        # print(face[0])
        # print(frameRGB.shape)
        Y_of_every_point = []
        X_of_every_point = []
        for origin in face:
            x , y, z = origin
            # cv2.circle(frame, (x, y), 1, color=(0,0,255), thickness=-1)
            cv2.circle(canvas, (x, y), 1, color=(0,0,255), thickness=-1)
            X_of_every_point.append(x)
            Y_of_every_point.append(y)
            


        # Finding minimum and maximun of x-coordinate and y-coordinate
        minX= maxX =X_of_every_point[0]
        minY= maxY =Y_of_every_point[0]
        
        for i in range(467): #467 is the total numbers of points on face

            if minX > X_of_every_point[i+1]:
                minX = X_of_every_point[i+1]
            if maxX < X_of_every_point[i+1]:
                maxX = X_of_every_point[i+1]
            
            if minY > Y_of_every_point[i+1]:
                minY = Y_of_every_point[i+1]
            if maxY < Y_of_every_point[i+1]:
                maxY = Y_of_every_point[i+1]
                       

        
        faceLocation.append((minX,minY))
        faceLocation.append((maxX,maxY))
        facesLocations.append(faceLocation)
        
        
    point += 1
            
    # if face_result.detections:
    #     for detection in face_result.detections:
    #         mp_draw.draw_detection(frame, detection)


    # checking if faces are there or not in an image
    if  len(facesLocations) != 0 :
        Number_of_faces = 0
        for faceLocation in facesLocations:

            x1,y1 = faceLocation[0]
            x2,y2 = faceLocation[1]

            if 0 <= x1-20 and x2+20 <= frame_width and 0 <= y1-90 and y2+20 <= frame_height:
                cv2.imshow("Croped "+str(Number_of_faces) + " face", frameCopy[y1-90:y2+20,x1-20:x2+20,:])
            # cv2.destroyWindow("Croped "+str(Number_of_faces))
            Number_of_faces +=1

    cv2.putText(frame, str(fps)+" fps",(20,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2, cv2.LINE_AA)
    cv2.imshow("Window", frame)
    canvas = cv2.resize(canvas, (0,0), dst=-1, fx=2, fy=2)
    cv2.imshow("Canvas", canvas)
    # cv2.imshow("canvas", canvas)
    # canvas[:,:,:] = 0
    cv2.waitKey(1)


cap.release()
cap.destroyAllWindows()


