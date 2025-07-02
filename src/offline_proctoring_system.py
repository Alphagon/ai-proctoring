import os
import sys
import cv2
import sys
import wget
import dlib
import argparse
from datetime import datetime

import numpy as np
from math import hypot
from collections import Counter
from db.postgresql import engine, extract_unproctored_record, Session, update_proctored_record

import face_recognition
from detection.misc import clean_up_directory, download_files, parse_video_path
from detection.headpose_estimation import load_hp_model
from detection.face_detection import get_face_detector, find_faces
from detection.custom_detection import *

################################################ Setup  ######################################################

def offline_proctoring(debug):
    # video file extensions
    video_extensions = ['.mp4', 'webm']

    # Attendee Face Encodings
    l = os.listdir('attendee_db')
    known_face_encodings = []
    known_face_names = []

    for file in l:
        if any(file.endswith(ext) for ext in video_extensions):
            video_path = f'attendee_db/{file}'
            print(video_path)
        else:
            attendee_image = face_recognition.load_image_file('attendee_db/' + file)
            attendee_face_encoding = face_recognition.face_encodings(attendee_image)[0]
            known_face_encodings.append(attendee_face_encoding)
            known_face_names.append(file.split('.')[0])

    # Headpose Model
    h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

    # Face Detection Model
    face_model = get_face_detector(modelFile='models/res10_300x300_ssd_iter_140000.caffemodel', configFile='models/deploy.prototxt')

    # Face Landmarks Model
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    # Video Capture
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) 
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
    DEBUG = debug

    # Desired output window size
    if DEBUG:
        output_width = 620
    else:
        output_width = 310
    output_height = int(output_width / aspect_ratio)

    #################################################### MAIN #####################################################

    while True:
        # Grabbing a frame of video
        ret, frame = video_capture.read()
        frame_count += 1
        # if frame_count <= 4000: continue
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
        if frame_count % 10 == 0:
            # Functionalities
            try:
                ##### Object Detection #####
                try:
                    count_items = get_objects_count(frame.copy())
                except Exception as error:
                    count_items = get_objects_count_exception()
                    print(error)

                #### Multiple People Functionality ####
                people_detection_frames, MULTIPLE_PEOPLE = people_detection(count_items, people_detection_frames, frame_count, fps, report, debug=DEBUG)

                #### Banned Object Detection Functionality #### 
                banned_object_frames, BANNED_OBJECTS = banned_object_detection(count_items, banned_object_frames, frame_count, fps, report, debug=DEBUG)

                # Checking Face detection/Face Verification/Headpose/Eye tracker details if and only if there is one person
                if count_items['person'] > 0:
                    #### Face Detection using caffe model of OpenCV's DNN module ####
                    #### Detecting Faces #### 
                    faces = find_faces(small_frame, face_model)

                    if len(faces) == 1:
                        face = faces[0]
                    else:
                        face_detection_frames = face_detection_offline(faces, face_detection_frames, frame_count, fps, report, debug=DEBUG)
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
                    face_verification_frames, FACE_VERIFICATION = face_verification(name, face_verification_frames, frame_count, fps, report, debug=DEBUG)

                    # Get Facial Landmarks
                    facial_landmarks = get_facial_landmarks(predictor, face, frame)

                    #### Headpose Functionality####
                    headpose_detection_frames, frame3, headpose_condition, HEADPOSE_DETECTION = head_pose_detection(
                        h_model, frame2, frame3, face, headpose_detection_frames, frame_count, fps, report, debug=DEBUG
                    )

                    ##### Eye Tracking Functionality#####
                    eye_tracking_frames, EYE_TRACKING = eye_tracker(frame2, facial_landmarks, eye_tracking_frames, headpose_condition, frame_count, fps, report, debug=DEBUG)
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

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

    return [MULTIPLE_PEOPLE, BANNED_OBJECTS, FACE_VERIFICATION, HEADPOSE_DETECTION, EYE_TRACKING]

def process_video_for_proctoring(engine, debug):
    with Session(engine) as session:
        candidate = extract_unproctored_record(session)
        if candidate is None: 
            print("All candidates are proctored")
            return

        # print(f"Proctoring {candidate}")
        video_filename, image_filename = download_files(candidate)
        updated_info = offline_proctoring(debug)
        update_proctored_record(session, candidate, updated_info)
        session.commit()  # Commit after all changes are done

    clean_up_directory(directory="src/attendee_db/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for proctoring.")
    # parser.add_argument("--video_path", type=parse_video_path, help="Path to the video file. Example: '/path/to/video.mp4' or you can use 0 for live")
    parser.add_argument("--debug", default=False, type=parse_video_path, help="Set it to True to enable debug mode (e.g., display output). Default is False.")
    args = parser.parse_args()
    print(args)
    process_video_for_proctoring(engine, debug=args.debug)