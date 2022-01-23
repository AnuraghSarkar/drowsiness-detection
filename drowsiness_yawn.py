# importing required modules
from threading import Thread
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist
import time
from twilio.rest import Client
import keys
import imutils

class EAR(object):
    @classmethod
    # function to calculate eye aspect ratio
    def eye_aspect_ratio(self, eye):
        first = dist.euclidean(eye[1], eye[5])
        second = dist.euclidean(eye[2], eye[4])
        third = dist.euclidean(eye[0], eye[3])
        return (first + second) / (2.0 * third)

    @classmethod
    # function to calculate EAR
    def final_ear(self, shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        ear = (leftEAR+rightEAR) /2.0
        return (ear, leftEye, rightEye)

    
    @classmethod
    # function to calculate lip distance
    def lip_distance(self, shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_lip_mean = np.mean(top_lip, axis=0)
        low_lip_mean = np.mean(low_lip, axis=0)
        lip_distance = abs(top_lip_mean[1] - low_lip_mean[1])
        return lip_distance

class SleepYawn(object):
    # initialize variables
    @classmethod
    def __init__(self):
        self.eye_aspect_ratio_threshold = 0.3
        self.eye_ar_consec_frames = 35
        self.yawn_threshold = 20
        self.counter = 0
        self.face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    
    @classmethod
    # function to send sms
    def send_sms(self, message):
        client = Client(keys.account_sid, keys.auth_token)

        message = client.messages.create(
            body=message,
            from_=keys.twilio_number,
            to=keys.target_number
        )

        print(message.body)

    # sound alarm
    @classmethod
    def buzzer(self):
        pass


    @classmethod
    # start video stream
    def start_video(self):
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        
        # start video stream thread
        print("[INFO] starting video stream thread...")
        self.vs = VideoStream().start()
        time.sleep(1.0)
        EAR_Obj = EAR()

        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            face_rectangle = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in face_rectangle:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            for face in faces:
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                    # EAR
                self.eye = EAR_Obj.final_ear(shape)
                self.ear = self.eye[0]
                self.leftEye = self.eye[1]
                self.rightEye = self.eye[2]
                    # lip distance
                lip_distance = EAR_Obj.lip_distance(shape)
                self.leftEyeHull = cv2.convexHull(self.leftEye)
                self.rightEyeHull = cv2.convexHull(self.rightEye)
                cv2.drawContours(frame, [self.leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [self.rightEyeHull],-1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if self.ear < self.eye_aspect_ratio_threshold:
                    self.counter += 1
                    if self.counter >= self.eye_ar_consec_frames:

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);cv2.putText(frame, "***********ALERT!***********", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        #self.send_sms("You are drowsing. Wash Your Face.")
                else:
                    self.counter = 0
                if lip_distance > self.yawn_threshold:
                    cv2.putText(frame, "YAWNING ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);cv2.putText(frame, "***********ALERT!***********", (10,325),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                   
                    #self.send_sms("You are yawning. Please get some air.")
                    # display ear and lip distance
                cv2.putText(frame, "EAR: {:.2f}".format(self.ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Lip: {:.2f}".format(lip_distance), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # display frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()
                    self.vs.release()
                    break
                
if __name__ == "__main__":
    sleep_yawn = SleepYawn()
    try:
        sleep_yawn.start_video()
    except Exception as e:
        print(e)
                    
