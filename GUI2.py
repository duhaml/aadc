
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

def drawHough(lines, output_image):
    drawnHough_image = np.copy(output_image)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(drawnHough_image,(x1,y1),(x2,y2),color=(80, 127, 0), thickness=5)

def lane_following_display(stage = 'all', duration=np.inf, saveVideo=False):
    """
    :param stage: 1) 'filter_colors'
                  2) 'grayscale'
                  3) 'canny'
                  4) 'region_of interest'
                  5) 'Canny'
                  6) 'Hough'
                  7) 'lane'
                  8) 'vanishing point'

                  np.inf) 'all'
                  0) 'none'
    :param duration:
    :param saveVideo:
    :return:
    """
    for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
        image = frame.array  #grab the raw numpy array representing the image

        ## core of line detection process

        #white and yellow filtering
        if stage == 'filter_colors' :
            color_filtered_image = ld.filter_colors(image)

            cv2.imshow("Frame", color_filtered_image)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue


        #grayscale the image
        if stage == 'grayscale':
            color_filtered_image = ld.filter_colors(image)

            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in imshow

            cv2.imshow("Frame", gray)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue


        #region of interest bounds:
        if stage == 'region_of interest':
            color_filtered_image = ld.filter_colors(image)
            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image

            vertices = ld.get_ROI_vertices(image)
            drawnROI_image = np.copy(gray)
            cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)

            cv2.imshow("frame", drawnROI_image)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue

        #edge detection:
        if stage == 'canny':
            color_filtered_image = ld.filter_colors(image)
            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image
            vertices = ld.get_ROI_vertices(image)
            drawnROI_image = np.copy(gray)
            cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
            ROI_image = ld.region_of_interest(image, vertices)

            canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
            image_cannyAndDrawnROI = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)  #pas return juste canny mais superposé au reste en tres transparent

            cv2.imshow('frame', image_cannyAndDrawnROI)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue

        #lane:
        if stage == 'Hough':
            color_filtered_image = ld.filter_colors(image)
            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image and ROI_image
            vertices = ld.get_ROI_vertices(image)
            drawnROI_image = np.copy(gray)
            cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
            ROI_image = ld.region_of_interest(gray, vertices)
            canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
            image_cannyAndDrawnROI = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)  #pas return juste canny mais superposé au reste en tres transparent

            lines = ld.hough_lines(canny_image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
            drawnHough_image = drawHough(lines, image_cannyAndDrawnROI)
            drawnHough_image_combined = cv2.addWeighted(drawnHough_image, 0.9, image_cannyAndDrawnROI, 0.1, 0.)

            cv2.imshow("frame", drawnHough_image_combined)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue

        if stage == 'lane':
           color_filtered_image = ld.filter_colors(image)
           gray = ld.grayscale(color_filtered_image)
           #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image and ROI_image
           vertices = ld.get_ROI_vertices(image)
           drawnROI_image = np.copy(gray)
           cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
           ROI_image = ld.region_of_interest(gray, vertices)
           canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
           image_cannyAndDrawnROI = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)  #pas return juste canny mais superposé au reste en tres transparent

           lines = ld.hough_lines(canny_image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
           drawnHough_image = drawHough(lines, image_cannyAndDrawnROI)
           drawnHough_image_combined = cv2.addWeighted(drawnHough_image, 0.9, image_cannyAndDrawnROI, 0.1, 0.)

           image_height = image.shape[0]
           right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
           #wtf is happening
           y2 = image_height * (1 - ld.trap_height)
           right_x1 = int((image_height - right_b) / right_m)
           right_x2 = int((y2 - right_b) / right_m)
           left_x1 = int((image_height - left_b) / left_m)
           left_x2 = int((y2 - left_b) / left_m)

           drawnLane_image=np.copy(drawnHough_image_combined)
           cv2.line(drawnLane_image, (right_x1, image_height), (right_x2, y2), color=(0, 69, 255), thickness=5)  #en transparent !
           cv2.line(drawnLane_image, (left_x1, image_height), (left_x2, y2), color=(0, 69, 255), thickness=5)

           drawnLane_image_combined = cv2. addWeighted(drawnLane_image, 0.9, image, 0.1, 0.)

           cv2.imshow('frame', drawnLane_image_combined)

           key = cv2.waitKey(1) & 0xFF
           rawCapture.truncate(0)  #clear the system in preparation for the next frame
           if key == ord("q"): #break the loop when 'q' key is pressed
               break
           continue



        ## core of lane following processes:



        #vanishing point:
        if stage == 'vanishing point':
            color_filtered_image = ld.filter_colors(image)
            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image and ROI_image
            vertices = ld.get_ROI_vertices(image)
            drawnROI_image = np.copy(gray)
            cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
            ROI_image = ld.region_of_interest(gray, vertices)
            canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
            image_cannyAndDrawnROI = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)  #pas return juste canny mais superposé au reste en tres transparent
            lines = ld.hough_lines(canny_image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
            drawnHough_image = drawHough(lines, image_cannyAndDrawnROI)
            drawnHough_image_combined = cv2.addWeighted(drawnHough_image, 0.9, image_cannyAndDrawnROI, 0.1, 0.)
            image_height = image.shape[0]
            right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
            #wtf is happening
            y2 = image_height * (1 - ld.trap_height)
            right_x1 = int((image_height - right_b) / right_m)
            right_x2 = int((y2 - right_b) / right_m)
            left_x1 = int((image_height - left_b) / left_m)
            left_x2 = int((y2 - left_b) / left_m)
            drawnLane_image=np.copy(drawnHough_image_combined)
            cv2.line(drawnLane_image, (right_x1, image_height), (right_x2, y2), color=(0, 69, 255), thickness=5)  #en transparent !
            cv2.line(drawnLane_image, (left_x1, image_height), (left_x2, y2), color=(0, 69, 255), thickness=5)
            drawnLane_image_combined = cv2. addWeighted(drawnLane_image, 0.9, image, 0.1, 0.)

            right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
            vanishingPoint = lf.vanishingPoint(right_m, right_b, left_m, left_b)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(drawnLane_image_combined, 'coordinates : '+vanishingPoint, org=(10,10), fontFace=font, fontScale=1, color=(0, 0, 255))

            if (0 <= vanishingPoint[0] < camera.resolution[0]) and (0 <= vanishingPoint[1] < camera.resolution[1]) :  # if the point is on the image
                center = ( int(vanishingPoint[0]), int(vanishingPoint[1]) ) #make sure its coordinates are int
                cv2.circle(drawnLane_image_combined, center, radius=7, color=(0, 0, 255), thickness=5)  # draw a circle around it

            drawnLane_image_refreshed = cv2.addWeighted(image, 0.6, drawnLane_image_combined, 0.4, 0.)
            cv2.imshow('Frame',drawnLane_image_refreshed)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break
            continue




        if stage == 'all':
            color_filtered_image = ld.filter_colors(image)
            gray = ld.grayscale(color_filtered_image)
            #imageUInt8 = np.uint8(gray) if needed, then change in drawnROI_image and ROI_image
            vertices = ld.get_ROI_vertices(image)
            drawnROI_image = np.copy(gray)
            cv2.polylines(drawnROI_image, [vertices], isClosed=True, color=(80, 127, 255), thickness=5)
            ROI_image = ld.region_of_interest(gray, vertices)
            canny_image = ld.canny(ROI_image, ld.low_threshold, ld.high_threshold)  # normalement ne va pas détecter les bordures de la ROI, et tout est noir, mais à voir
            image_cannyAndDrawnROI = cv2.addWeighted(canny_image, 0.9, drawnROI_image, 0.1, 0.)  #pas return juste canny mais superposé au reste en tres transparent
            lines = ld.hough_lines(canny_image, ld.rho, ld.theta, ld.threshold, ld.min_line_length, ld.max_line_gap)
            drawnHough_image = drawHough(lines, image_cannyAndDrawnROI)
            drawnHough_image_combined = cv2.addWeighted(drawnHough_image, 0.9, image_cannyAndDrawnROI, 0.1, 0.)
            image_height = image.shape[0]
            right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
            #wtf is happening
            y2 = image_height * (1 - ld.trap_height)
            right_x1 = int((image_height - right_b) / right_m)
            right_x2 = int((y2 - right_b) / right_m)
            left_x1 = int((image_height - left_b) / left_m)
            left_x2 = int((y2 - left_b) / left_m)
            drawnLane_image=np.copy(drawnHough_image_combined)
            cv2.line(drawnLane_image, (right_x1, image_height), (right_x2, y2), color=(0, 69, 255), thickness=5)  #en transparent !
            cv2.line(drawnLane_image, (left_x1, image_height), (left_x2, y2), color=(0, 69, 255), thickness=5)
            drawnLane_image_combined = cv2. addWeighted(drawnLane_image, 0.9, image, 0.1, 0.)
            right_m, right_b, left_m, left_b = ld.get_line_equations(image, lines, ld.theshold)
            vanishingPoint = lf.vanishingPoint(right_m, right_b, left_m, left_b)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(drawnLane_image_combined, 'coordinates : '+vanishingPoint, org=(10,10), fontFace=font, fontScale=1, color=(0, 0, 255))
            if (0 <= vanishingPoint[0] < camera.resolution[0]) and (0 <= vanishingPoint[1] < camera.resolution[1]) :  # if the point is on the image
                center = ( int(vanishingPoint[0]), int(vanishingPoint[1]) ) #make sure its coordinates are int
                cv2.circle(drawnLane_image_combined, center, radius=7, color=(0, 0, 255), thickness=5)  # draw a circle around it
            drawnLane_image_refreshed = cv2.addWeighted(image, 0.6, drawnLane_image_combined, 0.4, 0.)

            theta=lf.direction(drawnLane_image_refreshed, vanishingPoint)
            cv2.putText(drawnLane_image_refreshed, theta, org=(50, 10), fontFace=font, color=(0, 0, 255))

            cv2.imshow('Frame', drawnLane_image_refreshed)

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break


        if stage=='none':
            font = cv2.FONT_HERSHEY_SIMPLEX

            center = (image.shape[1]/2, image.shape[0]/2)
            cv2.line(frame, center, (image.shape[1]/2, 0), color=(100, 100, 255), thickness=5)  #ligne de référence

            right_m, right_b, left_m, left_b = ld.lane_detector(frame)
            vanishingPoint = lf.vanishingPoint(right_m, right_b, left_m, left_b)
            theta = lf.lane_follower(frame)

            cv2.line(frame, center, vanishingPoint, color=(100, 100, 255), thickness=5)  #ligne de direction
            cv2.putText(frame, 'coordinates : '+vanishingPoint, org=(10,10), fontFace=font, fontScale=1, color=(0, 0, 255))
            cv2.putText(image, theta, org=(50, 10), fontFace=font, color=(0, 0, 255))


            cv2.imshow("Frame", frame)  #show the frame

            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)  #clear the system in preparation for the next frame
            if key == ord("q"):  #break the loop when 'q' key is pressed
                break

