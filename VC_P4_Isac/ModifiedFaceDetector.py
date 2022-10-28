import cv2
from imutils import face_utils
import numpy as np
import dlib

class FaceDetector:
    def __init__(self):

        # dlib hog+svm for face detection
        self.detectordlib = dlib.get_frontal_face_detector()

        # dlib facial landmarks
        p = "shape_predictor_68_face_landmarks.dat"
        self.predictor68 = dlib.shape_predictor(p)

    def getLargest(self, objects):
        if len(objects) < 1:
            return -1
        elif len(objects) == 1:
            return 0
        else:
            areas = [w * h for x, y, w, h in objects]
            return np.argmax(areas)

    def getLargestRect(self, objects):
        if len(objects) < 1:
            return -1
        elif len(objects) == 1:
            return 0
        else:
            areas = [(rect.right() - rect.left()) * (rect.bottom() - rect.top()) for (i, rect) in enumerate(objects)]
            return np.argmax(areas)


    def DetectLargestFaceEyesDLIB(self, img):

        faces = self.detectordlib(img, 0)

        # Makes us of the largest face
        iface = self.getLargestRect(faces)
        if iface >= 0:
            x = faces[iface].left()
            y = faces[iface].top()
            w = faces[iface].right() - faces[iface].left()
            h = faces[iface].bottom() - faces[iface].top()

            values = self.GetFacialLandmarks(img, faces[iface], 1)

            if values is not None:
                points, shape = values

                # average of coordinates around the eyes
                le = np.mean(shape[36:42], 0)
                re = np.mean(shape[42:48], 0)

                return [x, y, w, h], [le[0], le[1], re[0], re[1]], shape

            else:
                return [-1, -1, -1, -1], [], []

        else:
            return [-1, -1, -1, -1], [], []


    def GetFacialLandmarks(self, img,roi,model=0):

        if model == 0:
            points = self.predictor(img, roi)
        else:
            points = self.predictor68(img, roi)

        if points is not None:
            # ibujamos recuadro mayor
            shape = face_utils.shape_to_np(points)
            # for (x, y) in shape:
            #     cv2.circle(imagenRGB, (x, y), 2, (255, 255, 255), -1)

            left_eye_x = int(points.part(3).x - abs(points.part(3).x - points.part(2).x) / 2.)
            left_eye_y = int(points.part(3).y - abs(points.part(3).y - points.part(2).y) / 2.)
            right_eye_x = int(points.part(1).x + abs(points.part(1).x - points.part(0).x) / 2.)
            right_eye_y = int(points.part(1).y - abs(points.part(1).y - points.part(0).y) / 2.)

            nose_x = int(points.part(4).x)
            nose_y = int(points.part(4).y)

            if right_eye_x - left_eye_x == 0:
                return None
            m1 = (right_eye_y - left_eye_y) / (right_eye_x - left_eye_x)
            if m1 != 0:
                m2 = -1 / m1
                b2 = nose_y - m2 * nose_x
                x_c = ((nose_y + 1.5 * abs(points.part(3).x - points.part(2).x) / 2.) - b2) / m2
                y_c = m2 * x_c + b2

                b3 = y_c - m1 * x_c
                left_mouth_x = ((y_c + m1 * 2 * abs(points.part(3).x - points.part(2).x) / 2.) - b3) / m1
                left_mouth_y = m1 * left_mouth_x + b3
                right_mouth_x = ((y_c - m1 * 2 * abs(points.part(3).x - points.part(2).x) / 2.) - b3) / m1
                right_mouth_y = m1 * right_mouth_x + b3
            else:
                left_mouth_x = points.part(4).x + 1.5 * abs(points.part(3).x - points.part(2).x) / 2.
                left_mouth_y = nose_y + abs(points.part(3).x - points.part(2).x) / 2.
                right_mouth_x = (points.part(4).x) - 1.5 * abs(points.part(3).x - points.part(2).x) / 2.
                right_mouth_y = nose_y + abs(points.part(3).x - points.part(2).x) / 2.

            points = [left_eye_x, right_eye_x, nose_x, left_mouth_x, right_mouth_x, left_eye_y, right_eye_y, nose_y,
                      left_mouth_y, right_mouth_y]

            return [points], shape
        else:
            return None

    def SingleFaceEyesDetection(self, img):
        face, eyes, shape = self.DetectLargestFaceEyesDLIB(img)
        return face, eyes, shape

