'''Detect if a person is wearing a glass or not,
using this python program. It's the most easiest
approach I've been used here.

  

import dlib
import numpy as np
from threading import Thread
import cv2
import time

# a cv2 font type
font_1 = cv2.FONT_HERSHEY_SIMPLEX
# Declare dlib frontal face detector
detector = dlib.get_frontal_face_detector()


# This class handles the video stream
# captured from the WebCam
class VideoStream:
    def __init__(self, stream):
        self.video = cv2.VideoCapture(stream)
        # Setting the FPS for the video stream
        self.video.set(cv2.CAP_PROP_FPS, 60)

        if self.video.isOpened() is False:
            print("Can't accessing the webcam stream.")
            exit(0)

        self.grabbed, self.frame = self.video.read()
        self.stopped = True
        # Creating a thread
        self.thread = Thread(target=self.update)
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.video.read()
        self.video.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# Capturing video through the WebCam. 0 represents the
# default camera. You need to specify another number for
# any external camera
video_stream = VideoStream(stream=0)
video_stream.start()

while True:
    if video_stream.stopped is True:
        break
    else:
        # Reading the video frame
        frame = video_stream.read()
        # Convert the frame color-space to grayscale
        # There are more than 150 color-space to use
        # BGR = Blue, Green, Red
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        # Get the coordinates of detected face
        for i, face_rect in enumerate(rects):
            left = face_rect.left()
            top = face_rect.top()
            width = face_rect.right() - left
            height = face_rect.bottom() - top

            # Draw a rectangle around the detected face
            # Syntax: cv2.rectangle(image, start_point, end_point,
            #  color, thickness)
            cv2.rectangle(frame, (left, top), (left + width, top + height), \
                          (0, 255, 0), 2)
            # Draw a face name with the number.
            # Syntax: cv2.putText(image, text, origin, font,
            # fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            # For better look, lineType = cv.LINE_AA is recommended.
            cv2.putText(frame, f"Face {i + 1}", (left - 10, top - 10), \
                        font_1, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            '''Cropping an another frame from the detected face rectangle'''
            frame_crop = frame[top + 10:top + height - 100, left + 30: \
                                                            left + width - 20]

            # Show the cropped frame
            cv2.imshow("Cropped Frame", frame_crop)

            # Smoothing the cropped frame
            img_blur = cv2.GaussianBlur(np.array(frame_crop), (5, 5), \
                                        sigmaX=1.7, sigmaY=1.7)
            # Filterting the cropped frame through the canny filter
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
            # Show the Canny Sample of the frame: 'frame_cropped'
            cv2.imshow("Canny Filter", edges)

            # Center Strip
            edges_center = edges.T[(int(len(edges.T) / 2))]
            # 255 represents white edges. If any white edges are detected
            # in the desired place, it will show 'Glass is Present' message
            if 255 in edges_center:
                cv2.rectangle(frame, (left, top + height), (left + width, \
                                                            top + height + 40), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Glass is Present", (left + 10, \
                                                        top + height + 20), font_1, 0.65, (255, 255, 255), 2,
                            cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (left, top + height), (left + width, \
                                                            top + height + 40), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "No Glass", (left + 10, top + height + 20), \
                            font_1, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # delay for processing a frame
        delay = 0.04
        time.sleep(delay)

    # show result
    cv2.imshow("Result", frame)

    key = cv2.waitKey(1)
    # Press 'q' for stop the executing of the program
    if key == ord('q'):
        break
# Stop capturing video frames
video_stream.stop()
# closing all windows
cv2.destroyAllWindows()