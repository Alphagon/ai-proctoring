################################################ Import Libraries  ##########################################
import cv2
import sys
import os
import matplotlib
import numpy as np
from collections import Counter
import face_recognition
import dlib
from math import hypot

from object_detection import yoloV3Detect
from landmark_models import *
from face_spoofing import *
from headpose_estimation import *
from face_detection import get_face_detector, find_faces
from custom_detection import get_objects_count, get_objects_count_exception,\
     people_detection, banned_object_detection, face_detection_online, \
     comparing_faces, face_verification, get_facial_landmarks, \
     head_pose_detection, eye_tracker
################################################ Setup  ######################################################

# face recognition
l = os.listdir('attendee_db')
known_face_encodings = []
known_face_names = []

for image in l:
    attendee_image = face_recognition.load_image_file('attendee_db/'+image)
    attendee_face_encoding = face_recognition.face_encodings(attendee_image)[0]

    known_face_encodings.append(attendee_face_encoding)
    known_face_names.append(image.split('.')[0])


# headpose model
h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

# face detection model
face_model = get_face_detector()

# face landmark model
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Others
video_capture = cv2.VideoCapture(0)
process_this_frame = True
no_of_frames_0 = 0
no_of_frames_1 = 0
no_of_frames_2 = 0
no_of_frames_3 = 0
no_of_frames_4 = 0
no_of_frames_5 = 0
no_of_frames_6 = 0
no_of_frames_7 = 0
flag = True
DEBUG = True

#################################################### MAIN #####################################################

while True:
    # frame skipping to save time
    process_this_frame = not process_this_frame 

    # Grab a single frame of video
    ret, frame = video_capture.read()

    frame2 = frame.copy()
    frame3 = frame.copy()
    report = np.zeros((frame3.shape[0],frame3.shape[1], 3), np.uint8)
  
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) # For debugging

    # Functionalities
    if process_this_frame:
        try:
            ##### Object Detection #####
            try:
                count_items = get_objects_count(small_frame)
                print(f'{count_items}')
            except Exception as e:
                count_items = get_objects_count_exception()
                print(e)

            # Multiple People Buffer
            no_of_frames_0 = people_detection(count_items, no_of_frames_0, report, debug=DEBUG)        
            # Banned Object Detection Buffer
            print("entering banned object detection")
            no_of_frames_1 = banned_object_detection(count_items, no_of_frames_1, report, debug=DEBUG)
            # Checking Face detection/Face Verification/Headpose/Eye tracker details if and only if there is one person.
            print("Done with banned object detection")
            
            if(count_items['person'] == 1):
                #### face detection using caffe model of OpenCV's DNN module ####
                # detect faces
                faces = find_faces(small_frame, face_model)

                if len(faces) > 0:
                    face = faces[0]
                else:
                    no_of_frames_7 = face_detection_online(faces, no_of_frames_7, report, debug=DEBUG)
                    horizontalAppendedImg = np.hstack((frame3,report))
                    cv2.imshow("Proctoring_Window", horizontalAppendedImg)
                    continue
                
                # Display Face Detection
                if DEBUG:
                    (left, top,right,bottom) = face
                    cv2.rectangle(frame3, (left*4, top*4), (right*4, bottom*4), (0, 0, 255), 2)

                if(flag==True):
                    #### face verification using face_recognition library ####
                    name = comparing_faces(small_frame, face, known_face_names, known_face_encodings)
                    flag = False
                
                # Face Detection Buffer
                no_of_frames_2 = face_verification(name, no_of_frames_2, report, debug=DEBUG)

                # Get Facial Landmarks
                facial_landmarks = get_facial_landmarks(predictor, face, frame)

                #### Head Pose ####
                no_of_frames_5, frame3, headpose_condition = head_pose_detection(h_model, frame2, frame3, face, no_of_frames_5, report, debug=True)

                ##### Eye Tracking #####
                no_of_frames_3 = eye_tracker(frame2, facial_landmarks, no_of_frames_3, headpose_condition, report, debug=True)
            else:
                flag = True


            horizontalAppendedImg = np.hstack((frame3,report))
            cv2.imshow("Proctoring_Window", horizontalAppendedImg)

        except Exception as e:
            print(e) 
            flag = True
            report = np.zeros((frame3.shape[0],frame3.shape[1], 3), np.uint8)

            #final display frame
            horizontalAppendedImg = np.hstack((frame3,report))
            cv2.imshow("Proctoring_Window", horizontalAppendedImg)
            

    # Display the resulting image
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("closing window...")
        break
    
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()