# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
import sys
 
if len(sys.argv) > 1:
    debug_mode = sys.argv[1]
else:
    debug_mode = 'False'

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

# cascade is just an XML file that contains the data to detect faces
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# welcome new entered person
previous_faces = 0

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # Face detection works faster in gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run face recognition on smaller images to speedup the process
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.77777775,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # If new faces are found, welcome them, and save resized face image for training
    if len(faces) > previous_faces:
        welcome_text = 'Willkommen bei Elgato!'
        print welcome_text
        # os.system("./speech.sh " + welcome_text + " &")
        timestamp = str(int(time.time()))
        cv2.imwrite("training/face_" + timestamp + "_" + ".jpg", image)

        face_counter = 0
        for (x, y, w, h) in faces:

            # All faces must have the same size at the end, make the face area wider or higher
            # until they fit the target size
            target_face_width = 120
            target_face_height = 150
            target_face_ratio = float(target_face_width)/float(target_face_height)
            face_ratio = float(w)/float(h)

            if target_face_ratio < face_ratio:
                new_face_width = w
                new_face_height = int(w / target_face_ratio)
                shift = int(float(new_face_height - h)/2)
                y = y - shift
                h = h + 2*shift
            elif target_face_ratio > face_ratio:
                new_face_height = h
                new_face_width = int(h * target_face_ratio)
                shift = int(float(new_face_width - w)/2)
                x = x - shift
                w = w + 2*shift

            # Save images in gray, not RGB
            face_gray = gray[y:(y+h+1), x:(x+w+1)]
            final_face_gray = cv2.resize(face_gray, (target_face_width, target_face_height), interpolation = cv2.INTER_CUBIC)

            # If in debug mode, display result
            if debug_mode == 'debug':
                cv2.imshow(str(face_counter), final_face_gray)

            # save result
            cv2.imwrite("training/face_" + timestamp + "_" + str(face_counter) + ".jpg", final_face_gray)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_counter += 1

    previous_faces = len(faces)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    if debug_mode == 'debug':
        cv2.imshow('Video', image)

    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
