import firebase_admin
from firebase_admin import auth
from firebase_admin import credentials
from firebase_admin import db

class RealTimeDatabase:
    # it takes a path to json file and initializt an app and connect to database
    def __init__(self, pathToFirebaseAdminFile) -> None:
        cred = credentials.Certificate(pathToFirebaseAdminFile)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://second-hand-book-exchang-1cf57-default-rtdb.firebaseio.com/',
	    })



    def addStudent(self, databaseReferencePath, studentDetails):
        """ Add sutudent to database\n
        It required database reference path (at which position want to store data) and studennt data as JSON 
        """
        ref = db.reference(databaseReferencePath)
        ref.update(studentDetails)
    
    def getDataFromDatabase(self, databaseReferencePath):
        ref = db.reference(databaseReferencePath)
        return ref.get()
    
    def deleteDataBase(self, databaseReferencePath):
        """Delete a specfic node from database"""
        ref = db.reference(databaseReferencePath)
        ref.delete()

    
    