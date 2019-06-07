
from picamera.array import PiRGBArray
from picamera import PiCamera
import time, cv2
import numpy as np
from io import BytesIO

camera1 = PiCamera(camera_num=1)
camera0 = PiCamera(camera_num=0)
camera1.resolution = (640, 480)
camera0.resolution = (640, 480)
camera1.framerate = 32
camera0.framerate = 32
rawCapture = PiRGBArray(camera1, size = camera1.resolution)
rawCapture = PiRGBArray(camera0, size = camera0.resolution)

mystream1 = BytesIO

camera1.start_recording(mystream1)
camera1.wait_recording(100)
camera1.stop_recording()

mystream0 = BytesIO

camera0.start_recording(mystream0)
camera0.wait_recording(100)
camera0.stop_recording()


