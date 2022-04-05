import sys
sys.path.append("C:\\Users\\User\\Desktop\\Projests Practicing for self\\Image Processing\\Face Recognition Attendence System\\Get Student Details")
import GetStudentDetails.Get_Student_Details as Get_Student_Details 
import RealTimeDatabase
realTimeDatabase = RealTimeDatabase.RealTimeDatabase(pathToFirebaseAdminFile="second-hand-book-exchang-1cf57-firebase-adminsdk-ae2w6-5e934cc6a2.json")
get_student_details = Get_Student_Details.GetStudentDetails(realTimeDatabase=realTimeDatabase)




	
# ref = db.reference("/Image Capture/Mubashir")
# cap = cv2.VideoCapture(0)
# count = 0
# listFrame = []
# frameStr = ""
# while (True):
# 	succ, frame = cap.read()
# 	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	for i in frame:
# 		# listFrame.append(i)
# 		frameStr = frameStr + str(i)
# 		frameStr = frameStr + " "
# 	# JsonFram = json.dump(listFrame)
# 	ref.update({"frame update"+str(count):frameStr})
# 	count += 1

# 	cv2.imshow("frame",frame)
# 	cv2.waitKey(1)



isRegistration = input("Are you here to register as a student (Y / N)\n")
if isRegistration == "Y" or isRegistration == "y":
    # print(isRegistration)
	studentDetails = get_student_details.registerStudent()
	print("Below data is stored in database")
	print(studentDetails)
	del isRegistration

# databaseData = realTimeDatabase.getDataFromDatabase(databaseReferencePath="/")
# print(databaseData)

realTimeDatabase.addStudent(databaseReferencePath="/",
					studentDetails={studentDetails["UIN"]:studentDetails})