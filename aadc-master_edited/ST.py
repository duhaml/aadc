

from picamera.array import PiRGBArray
from picamera import PiCamera
import time, cv2
import numpy as np

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size = camera.resolution)

time.sleep(0.1)

for image in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    print(type(image))
