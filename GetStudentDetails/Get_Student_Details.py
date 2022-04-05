import time
import sys
import os
# RealTimeDatabase.py
sys.path.append('/home/mubashir/Desktop/Image Processing/Face Recognition Attendence System/')
import RealTimeDatabase
import cv2
import Face_Mesh.medoapipe_face_mesh as medoapipe_face_mesh
# realTimeDatabase = RealTimeDatabase.RealTimeDatabase(pathToFirebaseAdminFile="second-hand-book-exchang-1cf57-firebase-adminsdk-ae2w6-5e934cc6a2.json")

class GetStudentDetails:

    def __init__(self, realTimeDatabase) -> None:
        self.realTimeDatabase = realTimeDatabase

    # This function will call getStudentDetails() and take name, UIN of the student
    # and then start the camera for creating student face encoding 
    def registerStudent(self):
        studentDetails = self.getStudentDetails()
        return studentDetails
    
    def checkStudentIsPresentInDatabase(self,UIN):
        print("Checking Student is already present in Database or Not")
        studentDataIndatabase = {}
        studentDataIndatabase["UIN"] = self.realTimeDatabase.getDataFromDatabase("/"+str(UIN)+"/UIN")
        # studentDataIndatabase["name"] = self.realTimeDatabase.getDataFromDatabase("/"+str(UIN)+"/name")
        return studentDataIndatabase

    def createFolder(self,directory):
        
        try:
            if not os.path.exists(directory):
                print("making directory "+directory)
                os.makedirs(directory)
        except OSError:
            print("Error creating directory", OSError)  

    # This function get student details such as UIN, Name, webcam Number from keyboard 
    # and create a folder of student with UIN_name pattern if folder is not present 
    def getStudentDetails(self):
        year = input("Enter your year:\nFY\nSY\nTy\nBY\n").upper()
        branch = input("\nEnter your Engineering Branch: ").upper()
        branch = branch
        UIN = input("\nEnter your UIN: ")
        studentDataIndatabase = self.checkStudentIsPresentInDatabase(UIN=UIN)
        print(studentDataIndatabase)
        if studentDataIndatabase["UIN"] != None:
            print("student is already registered with Details\nUIN:",studentDataIndatabase["UIN"],"\nname: ")
            exit(0)
        name = input("\nEnter your name: ")
        # we will use this in future for to take video from a mp4 file instead of a webcam
        camNumber = input("\nEnter Camera Number as zero '0':   ")
        if camNumber != "0":
            print("!!!!!!!!!!!!! Please follow the instruction correctly !!!!!!!!!!!!\n")
            print("!!!!!!!!!!!! You were told to put camera number as zero'0' !!!!!!!!!!!!\n")
            camNumber = "0"


        studentFolderPath = "GetStudentDetails/Students Encoding of Face/"+str(UIN)+"_"+name
        studentDetails = {"year":year,"branch":branch,"UIN":UIN,"name":name,"camNumber":camNumber,"studentFolderPath":studentFolderPath, "isRegistered":False}
       
        # change directory to Get Stuent Details Folder
        
        # studentFolderPresent = os.listdir("Students Encoding of Face")
        # for i in studentFolderPresent:
        #     j = i.split("_")
        #     if j[0] == str(UIN):
        #         studentDetails["isRegistered"] = True
        #         break
        # os.chdir("./Get Student Details")
        if not studentDetails["isRegistered"]:
            if not os.path.exists(studentFolderPath):
                try:
                    print("About to create a student encoding floder and save 100 image of its face")
                    self.createFolder(studentFolderPath)
                    self.startCamera(studentDetails["camNumber"], studentDetails["UIN"], studentFolderPath)
                    studentDetails["isRegistered"] = True
                except Exception as e:
                    print("Unable to create a student encoding floder Error is:")
                    print(e)
            
        else:
            print("Student is already exists")
        return studentDetails

    # start camera to detect face
    def startCamera(self, camNumber, UIN, studentFolderPath):
        localMP = medoapipe_face_mesh.FaceMesh()
        cap = cv2.VideoCapture(int(camNumber))
        print("Stay in fromt of the camera")
        count = 0
        pastTime = 0
        while(True):
            success, frame = cap.read()
            if success != True or count > 100:
                print("Video has ended")
                break

            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


            # processing face mesh and storing all point in a list in pixel form
            # for processing face mesh the function required image in RGB format
            face_meshs_custom_landmarks = localMP.faceMeshDetection(frameRGB)


            frameRGB = cv2.cvtColor(frameRGB,cv2.COLOR_RGB2BGR)
            # frameRGB = localMP.drawFaceMeshPoints(frameRGB, face_meshs_custom_landmarks=face_meshs_custom_landmarks)

            faceLocation = localMP.extractFace(face_meshs_custom_landmarks=face_meshs_custom_landmarks)
            
            
            
            
            
            
            
            # If face is present in an Image the this will croped face and display in other window
            # and if face is out of frame then it will destroy the croped window
            if faceLocation:
                print("Croped Image 0")
                face1 = faceLocation[0]
                if face1[:] >=(0,0,0,0):
                    print("Croped Image 1")
                    print(face1)
                    cv2.imshow("Croped", frame[face1[0]:face1[2],face1[3]:face1[1],:3])

                    # print(current_working_dir)
                    # if current_working_dir.split("\\")[-1] != str(UIN)+"_"+name:
                    #     os.chdir(current_working_dir+"\\"+"Students Encoding of Face\\"+str(UIN)+"_"+name)
                    photoName =studentFolderPath +"/"+str(UIN)+"_"+str(count)+".png"
                    cv2.imwrite(photoName, frame[face1[0]:face1[2],face1[3]:face1[1],:3])    
                    
                    count+=1
            else:
                print("Croped Image 2")
                cv2.putText(frameRGB, "Face is not visible", (50,70), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,0,255), thickness=3)
                try:
                    cv2.destroyWindow(winname="Croped")
                except:
                    pass
            currentTime = time.time()
            fps = int(1/(currentTime - pastTime))
            pastTime = currentTime

            cv2.putText(frameRGB, str(fps)+" FPS", (30,30), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,120,255), thickness=2)
            cv2.imshow("frame", frameRGB)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        cap.release()


    


        