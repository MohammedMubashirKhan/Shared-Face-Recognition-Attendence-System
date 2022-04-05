# https://www.geeksforgeeks.org/python-multiple-face-recognition-using-dlib/

 
# importing libraries
import pickle
import cv2
import face_recognition
import docopt
from sklearn import svm
import os
import Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
Localmp = medoapipe_face_mesh.FaceMesh()
  
def face_recognize(dir, test):
    # Training the SVC classifier
    # The training data would be all the 
    # face encodings from all the known 
    # images and the labels are their names
    encodings = []
    names = []
    dir = "GetStudentDetails/Students Encoding of Face"
    # Training directory
    if dir[-1]!='/':
        dir += '/'
    train_dir = os.listdir(dir)
  
    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(dir + person)
  
        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = cv2.imread(
                dir + person + "/" + person_img)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_bounding_boxes = [(0, face.shape[1], face.shape[0], 0)]
            cv2.imshow("face", cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            for i in range(5000):
                pass
  
            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face, known_face_locations= face_bounding_boxes)[0]
                # Add face encoding for current image 
                # with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(person + "/" + person_img + " can't be used for training")
    
    print((encodings))
    print((names))
 
    # Create and train the SVC classifier
    clf = svm.SVC(gamma ='scale', probability=True, )
    clf.fit(encodings, names)
    with open("model.sav", "wb") as f:
        pickle.dump(clf, f)
  
    # Load the test image with unknown faces into a numpy array
    test_image = face_recognition.load_image_file("test1.jpeg")
    test_image = cv2.resize(test_image, dsize=(0,0), fx=0.8, fy=0.8)
  
    # Find all the faces in the test image using the default HOG-based model
    face_locations = face_recognition.face_locations(test_image)

    no = len(face_locations)
    print("Number of faces detected: ", no)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(test_image,(left,top), (right, bottom), color=(0,0,255), thickness=2)
        cv2.imshow("test image", cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
  
    # Predict all the faces in the test image using the trained classifier
    print("Found:")
    for i in range(no):
        test_image_enc = face_recognition.face_encodings(test_image, known_face_locations=face_locations)[i]
        name = clf.predict([test_image_enc])
        print(*name)
    del(clf)
    del(encodings)
    del(names)
    
  
def main():
    
    train_dir = "GetStudentDetails/Students Encoding of Face"
    test_image = "--test_image"
    face_recognize(train_dir, test_image)
  
if __name__=="__main__":
    main()