
ATTENTION = '''ORIGINE EN HAUT A GAUCHE ET ICI PARTI DU PRINCIPE QUE BAS A GAUCHE
de même pour ld et lf

'''



AF = '''
supprimer les variables intermédiaires ?
combiner les fonctions (tout afficher, quitte à passer en argument les aspects à afficher) parce que là on recalcule deux fois les lignes
ASPECTS à AFFICHER :
    1) 'filter_colors'
    2) 'canny'
    3) 'region_of interest'
    4) canny
    5) Hough
    
    np.inf) 'all'
    
duration

enregistrer pour ensuite faire du image par image
'''


toBeRanOnRasPi="""
basically we need to enable real time video processing on the RasPi via openCV, so :
$ sudo raspi-config     : option 5 enable camera; then reboot
$ source ~/.profile
$ workon cv
$ pip install "picamera[array]"
"""

from picamera.array import PiRGBArray
from picamera import PiCamera
import time, cv2
import numpy as np

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size = camera.resolution)

time.sleep(0.1)


# Lane detection:
import lane_detector as ld
import lane_follower as lf

def lane_following_display(stage = all, duration=np.inf, saveVideo=False):
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array  #grab the raw numpy array representing the image

        ## core of line detection process

        #white and yellow filtering
        color_filtered_image = ld.filter_colors(image)
        if stage == 'filter_colors' :
            cv2.imshow("Frame", color_filtered_image)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break

        #region of interest bounds:
        relative_bottom_width = ld.trap_bottom_width
        relative_top_width = ld.trap_top_width
        relative_height = ld.trap_height
        image_height = image.shape[0]
        image_width = image.shape[1]
        bottom_left_point = [(1-relative_bottom_width*image_width)/2, 0]
        bottom_right_point = [(1+relative_bottom_width*image_width)/2, 0]
        top_left_point = [(1-relative_top_width*image_width)/2, relative_height*image_height]
        top_right_point = [(1+relative_top_width*image_width)/2, relative_height*image_height]
        vertices = np.array([bottom_left_point, bottom_right_point, top_left_point, top_right_point],np.int32)
        vertices.reshape((-1, 1, 2))
        drawnROI_image = np.copy(image)
        cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
        if stage == 'region_of interest':
            cv2.imshow("frame", drawnROI_image)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
        ROI_image = ld.region_of_interest(image, vertices)

        #edge detection
        canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
        if stage == 'canny':
            image2 = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)
            cv2.imshow('frame', image2)  #pas return juste canny mais superposé au reste en tres transparent
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break

        #lane:
        lines = ld.hough_lines(image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
        right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
        y2 = image_height * (1 - relative_height)
        right_x1 = int((image_height - right_b) / right_m)
        right_x2 = int((y2 - right_b) / right_m)
        left_x1 = int((image_height - left_b) / left_m)
        left_x2 = int((y2 - left_b) / left_m)
        cv2.line(image, (right_x1, image_height), (right_x2, y2), color=(0, 69, 255), thickness=5)
        cv2.line(image, (left_x1, image_height), (left_x2, y2), color=(0, 69, 255), thickness=5)





        ## core of lane following processes:

        font = cv2.FONT_HERSHEY_SIMPLEX

        #vanishing point:
        lines = ld.hough_lines(image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
        right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
        vanishingPoint = lf.vanishingPoint(right_m, right_b, left_m, left_b)
        cv2.putText(image, 'coordinates : '+vanishingPoint, org=(10,10), fontFace=font, fontScale=1, color=(0, 0, 255))
        if (0 <= vanishingPoint[0] < camera.resolution[0]) and (0 <= vanishingPoint[1] < camera.resolution[1]) :  # if the point is on the image
            center = ( int(vanishingPoint[0]), int(vanishingPoint[1]) ) #make sure its coordinates are int
            cv2.circle(image, center, radius=7, color=(0, 0, 255), thickness=5)  # draw a circle around it




        if stage == 'all':
            cv2.imshow("Frame", image)  #show the frame
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break


                
                

  def lane_following_display(stage = all, duration=np.inf, saveVideo=False):
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array  #grab the raw numpy array representing the image

        ## core of line detection process

        #white and yellow filtering
        color_filtered_image = ld.filter_colors(image)
        if stage == 'filter_colors' :
            cv2.imshow("Frame", color_filtered_image)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break


        #edge detection
        canny_image = ld.canny(color_filtered_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
        if stage == 'canny':
            cv2.imshow('frame', canny_image)  #pas return juste canny mais superposé au reste en tres transparent
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
lane_following_display('canny')


def reste():
    #lane
        image2 = np.uint8(image)
        #cvtColor(dst, cdst, CV_GRAY2BGR)
        lines = ld.hough_lines(image2, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
        right_m, right_b, left_m, left_b = ld.get_line_equations(image2, lines, ld.theshold)
        y2 = image_height * (1 - relative_height)
        right_x1 = int((image_height - right_b) / right_m)
        right_x2 = int((y2 - right_b) / right_m)
        left_x1 = int((image_height - left_b) / left_m)
        left_x2 = int((y2 - left_b) / left_m)
        cv2.line(image, (right_x1, image_height), (right_x2, y2), color=(0, 69, 255), thickness=5)
        cv2.line(image, (left_x1, image_height), (left_x2, y2), color=(0, 69, 255), thickness=5)


