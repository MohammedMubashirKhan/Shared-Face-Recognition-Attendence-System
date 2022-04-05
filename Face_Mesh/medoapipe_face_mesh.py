import cv2
import mediapipe as mp


class FaceMesh:
    
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=10,static_image_mode=True)
        self.mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
        # self.mp_drawing_sty   les = mp.solutions.drawing_styles

            
    def faceMeshDetection (self, image, enlargeby=1):
        """
        It takes a frame of an image and return list of face meshs\n
        This function return a list of all face mesh
        """
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        face_mesh_result = self.face_mesh.process(image)

        frame_height = image.shape[0]
        frame_width = image.shape[1]
        face_meshs_custom_landmarks=[]

        # if face_mesh_result.multi_face_landmarks:
        #     for face_landmarks in face_mesh_result.multi_face_landmarks:

        #         self.mp_drawing.draw_landmarks(
        #     image=image,
        #         landmark_list=face_landmarks,
        #       connections=self.mp_face_mesh.FACEMESH_CONTOURS,
        #       landmark_drawing_spec=None,)
            #   connection_drawing_spec=mp_drawing_styles
            #   .get_default_face_mesh_contours_style())



        if face_mesh_result.multi_face_landmarks:
            for face_landmarks in face_mesh_result.multi_face_landmarks:
                
                face_mesh_custom_landmark = []
                for landmark in face_landmarks.landmark:
                    x = landmark.x * frame_width
                    x = round (x) *enlargeby
                    y = landmark.y * frame_height
                    y = round (y) *enlargeby
                    z = landmark.z *enlargeby
                    face_mesh_custom_landmark.append((x, y, z))
                face_meshs_custom_landmarks.append(face_mesh_custom_landmark)
                # print((face_meshs_custom_landmarks[0][0][0]))
        return face_meshs_custom_landmarks
    

    # Drawing all point on a face in an image provided to function
    def drawFaceMeshPoints(self, image, face_meshs_custom_landmarks):
  
        for face in face_meshs_custom_landmarks:
            for origin in face:
                x, y, z = origin
                cv2.circle(image, (x, y), 1, color=(0,0,255), thickness=-1)        
        return image

    # Return face location in a list (xmin, ymin, xmax, ymax)
    def extractFace(self, face_meshs_custom_landmarks, enlargeby=1):
        faecLocation = []
        for face in face_meshs_custom_landmarks:
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0
            for origin in face:
                x, y, z = origin
                if(xmin == 0 and xmax == 0):
                    xmin = x
                    xmax = x
                    ymin = y
                    ymax = y

                if xmin > x and x >= 0:
                    xmin = x
                if ymin > y and y >= 0:
                    ymin = y
                if xmax < x :
                    xmax = x
                if ymax < y :
                    ymax = y
            # xmin *= enlargeby
            # xmax *= enlargeby
            # ymin *= enlargeby
            # ymax *= enlargeby
            faecLocation.append((ymin ,xmax,ymax,xmin))
        return faecLocation
                
                    


        
        
