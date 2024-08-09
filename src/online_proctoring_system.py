################################################ Import Libraries  ##########################################
import os
import sys
import cv2
import dlib
import matplotlib
import numpy as np
from math import hypot
import face_recognition
from collections import Counter
import argparse  # Import argparse for command-line argument parsing

from detection.headpose_estimation import load_hp_model
from detection.face_detection import get_face_detector, find_faces
from detection.custom_detection import (
    get_objects_count,
    get_objects_count_exception,
    people_detection,
    banned_object_detection,
    face_detection_online,
    comparing_faces,
    face_verification,
    get_facial_landmarks,
    head_pose_detection,
    eye_tracker,
)

################################################ Setup  ######################################################

def main(video_path):
    # Attendee Face Encodings
    l = os.listdir('attendee_db')
    known_face_encodings = []
    known_face_names = []

    for image in l:
        attendee_image = face_recognition.load_image_file('attendee_db/' + image)
        attendee_face_encoding = face_recognition.face_encodings(attendee_image)[0]

        known_face_encodings.append(attendee_face_encoding)
        known_face_names.append(image.split('.')[0])

    # Headpose Model
    h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

    # Face Detection Model
    face_model = get_face_detector(modelFile='models/res10_300x300_ssd_iter_140000.caffemodel', configFile='models/deploy.prototxt')

    # Face Landmarks Model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Video Capture
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = video_width / video_height

    people_detection_frames = 0
    banned_object_frames = 0
    face_verification_frames = 0
    headpose_detection_frames = 0
    eye_tracking_frames = 0
    face_detection_frames = 0
    flag = True
    DEBUG = False

    # Desired output window size
    output_width = 620
    output_height = int(output_width / aspect_ratio)

    print(frame_count)

    #################################################### MAIN #####################################################

    while True:
        # Grabbing a frame of video
        ret, frame = video_capture.read()
        frame_count += 1

        if not ret:
            print("End of video")
            break

        frame = cv2.resize(frame, (output_width, output_height))
        frame2 = frame.copy()
        frame3 = frame.copy()
        report = np.zeros((frame3.shape[0], frame3.shape[1], 3), np.uint8)

        # Resize frame to 1/5th for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

        # Frame-Skipping to save time
        if frame_count % 5 == 0:
            # Functionalities
            print(frame_count)
            try:
                ##### Object Detection #####
                try:
                    count_items = get_objects_count(frame.copy())
                except Exception as error:
                    count_items = get_objects_count_exception()
                    print(error)

                #### Multiple People Functionality ####
                people_detection_frames = people_detection(count_items, people_detection_frames, frame_count, fps, report, debug=DEBUG)

                #### Banned Object Detection Functionality #### 
                banned_object_frames = banned_object_detection(count_items, banned_object_frames, frame_count, fps, report, debug=DEBUG)

                # Checking Face detection/Face Verification/Headpose/Eye tracker details if and only if there is one person
                if count_items['person'] > 0:
                    #### Face Detection using caffe model of OpenCV's DNN module ####
                    #### Detecting Faces #### 
                    faces = find_faces(small_frame, face_model)

                    if len(faces) == 1:
                        face = faces[0]
                    else:
                        face_detection_frames = face_detection_online(faces, face_detection_frames, frame_count, fps, report, debug=DEBUG)
                        if DEBUG:
                            horizontalAppendedImg = np.hstack((frame3, report))
                            cv2.imshow("Proctoring_Window", horizontalAppendedImg)
                        continue
                    
                    # Display Detected Face
                    if DEBUG:
                        (left, top, right, bottom) = face
                        cv2.rectangle(frame3, (left * 5, top * 5), (right * 5, bottom * 5), (0, 0, 255), 2)

                    if flag:
                        #### Face verification using face_recognition library ####
                        name = comparing_faces(small_frame, face, known_face_names, known_face_encodings)
                        flag = False
                    
                    #### Face Verification Functionality #### 
                    face_verification_frames = face_verification(name, face_verification_frames, frame_count, fps, report, debug=DEBUG)

                    # Get Facial Landmarks
                    facial_landmarks = get_facial_landmarks(predictor, face, frame)

                    #### Headpose Functionality####
                    headpose_detection_frames, frame3, headpose_condition = head_pose_detection(
                        h_model, frame2, frame3, face, headpose_detection_frames, frame_count, fps, report, debug=DEBUG
                    )

                    ##### Eye Tracking Functionality#####
                    eye_tracking_frames = eye_tracker(frame2, facial_landmarks, eye_tracking_frames, headpose_condition, frame_count, fps, report, debug=DEBUG)
                else:
                    flag = True
                if DEBUG:
                    horizontalAppendedImg = np.hstack((frame3, report))
                    cv2.imshow("Proctoring_Window", horizontalAppendedImg)

            except Exception as e:
                print(e) 
                flag = True
                report = np.zeros((frame3.shape[0], frame3.shape[1], 3), np.uint8)

                # Final display frame
                if DEBUG:
                    horizontalAppendedImg = np.hstack((frame3, report))
                    cv2.imshow("Proctoring_Window", horizontalAppendedImg)

        # Display the resulting image
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("closing window...")
            break

    print(frame_count)

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for proctoring.")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    args = parser.parse_args()
    
    main(args.video_path)