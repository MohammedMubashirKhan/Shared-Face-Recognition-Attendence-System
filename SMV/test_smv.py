import pickle
import time
import face_recognition
import cv2
import Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
localMP = medoapipe_face_mesh.FaceMesh()
import Face_Mesh.mediapipe_face_detection as mediapipe_face_detection
mp_faceDetection = mediapipe_face_detection.FaceDetection()

with open("SMV/model.sav", "rb") as f:
    clf = pickle.load(f)

cap = cv2.VideoCapture(0)


pastTime = 0
while(True):
    success, frame = cap.read()
    if success != True or cv2.waitKey(1) == ord('q'):
        print("Video has ended")
        break

    # Load the test image with unknown faces into a numpy array
    test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # test_image = cv2.resize(test_image, dsize=(0,0), fx=0.25, fy=0.25)


    ## processing face mesh and storing all point in a list in pixel form
    ## for processing face mesh the function required image in RGB format
    face_meshs_custom_landmarks = localMP.faceMeshDetection(test_image, )
    # localMP.drawFaceMeshPoints(frame, face_meshs_custom_landmarks=face_meshs_custom_landmarks)
    face_locations = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)



    # Find all the faces in the test image using the default HOG-based model
    # face_locations = face_recognition.face_locations(test_image)

    no = len(face_locations)
    # print("Number of faces detected: ", no)
    if no == 0:
        continue
    else:
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(test_image,(left,top), (right, bottom), color=(0,0,255), thickness=2)
            # cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
            cv2.imshow("test image", test_image)
            # cv2.waitKey(0)

    currentTime = time.time()
    fps = 1/(currentTime - pastTime)
    fps = int(fps)
    cv2.putText(test_image, str(fps)+" FPS", (left,top-15), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(255,0,0))

    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image, known_face_locations=face_locations)[i]
        test_image_enc1 = face_recognition.face_encodings(test_image, known_face_locations=face_locations)[i]
        name = clf.predict([test_image_enc])
        clf.fit([test_image_enc,test_image_enc1], ["Mubashir","Mubashir1"])
        print(*name)
        (top, right, bottom, left) = face_locations[i]
        cv2.putText(test_image, str(*name), (left,top+15), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=1,
                color=(255,0,0))
        # test_image = cv2.resize(test_image, dsize=(0,0), fx=4, fy=4)
        cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        # print(*name)

pastTime =time.time()
# Load the test image with unknown faces into a numpy array
test_image = cv2.imread("SMV/test1.jpeg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, dsize=(0,0), fx=0.8, fy=0.8)
cv2.imshow("test image",test_image)
mp_faceDetection.faceDetection(test_image)


## processing face mesh and storing all point in a list in pixel form
## for processing face mesh the function required image in RGB format
face_meshs_custom_landmarks = localMP.faceMeshDetection(test_image,)
# print(face_meshs_custom_landmarks)
# localMP.drawFaceMeshPoints(frame, face_meshs_custom_landmarks=face_meshs_custom_landmarks)
face_locations = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)



# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)

no = len(face_locations)
print("Number of faces detected: ", no)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(test_image,(left,top), (right, bottom), color=(0,0,255), thickness=2)
    # cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    cv2.imshow("test image", test_image)
    # cv2.waitKey(0)

# cv2.putText(test_image, str(fps)+" FPS", (left,top-15), 
#             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale=1,
#             thickness=2,
#             color=(255,0,0))

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image, known_face_locations=face_locations)[i]
    name = clf.predict([test_image_enc])
    print(*name)
    (top, right, bottom, left) = face_locations[i]
    cv2.putText(test_image, str(*name), (left,top+15), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            thickness=1,
            color=(255,0,0))
    # gets a dictionary of {'class_name': probability}
    prob_per_class_dictionary = dict(zip(clf.classes_, name))

    # gets a list of ['most_probable_class', 'second_most_probable_class', ..., 'least_class']
    results_ordered_by_probability = map(lambda x: x[0], sorted(zip(clf.classes_, name), key=lambda x: x[1], reverse=False))
    print(clf.classes_," MOST probablr Class")
    results_ordered_by_probability = clf.predict_proba([test_image_enc])
    print(results_ordered_by_probability)
    # fps = int(fps)

    cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    # print(*name)
currentTime = time.time()
fps = 1/(currentTime - pastTime)
cv2.putText(test_image, str(fps)+" FPS", (50,70), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            thickness=2,
            color=(255,0,0))

cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
