'''
1) Keep only yellow and white pixels, black out all other pixels: filter_colors()
    This removes any unwanted edges from shadows, cracks in the road, etc, BUT MIGHT NOT WORK WHEN IT RAINS
2) Convert image into grayscale: grayscale()
3) Apply Gaussian smoothing: gaussian_blur()
4) Run Canny Edge Detector: canny()
5) Create a trapezoidal region-of-interest in the center lower-half of the image: region_of_interest()
6) Run Hough Line Detector: hough_lines()
7) Filter Hough lines by slope and endpoint location, and separate them into candidate right/left lane line segments.
8) run linear regression on candidate right/left lane line segment endpoints, to create right/left lane line equations.
9) From these equations, compute the equations of the right/left lane lines : get_equations()
'''


import numpy as np
import cv2

## Global parameters


# color filtering
white_threshold = 200 #130      not enough white detected : more contrast, better than adjusting luminosity

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 0.85  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.07  # idem for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height


# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments

# line selection
slope_threshold = 0.5


# colors are in BGR code


# Helper functions

def filter_colors(image):
    """
    Filter the image to include only yellow and white pixels.
    /!\ affects the original image
    """
    # Filter white pixels
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(image, lower_white, upper_white)
    white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #lower_yellow = np.array([90,100,100])
        #upper_yellow = np.array([110,255,255])
        #yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        #yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

        #image2 = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.) # Combine the two above images

    return white_image


def grayscale(image):
    """Applies a greyscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(image, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    """Applies the Canny transform : only keeps the edges"""
    return cv2.Canny(image, low_threshold, high_threshold)


def get_ROI_vertices(image, thêta=0):
    """
    :param image:
    :param thêta: the angle returned by lane follower (eventually the ROI will adapt to the road)
    :return: an array containing the 4 vertices
    """
    image_height = image.shape[0]
    image_width = image.shape[1]

    bottom_left_point = [(1-trap_bottom_width)*image_width/2, image_height]
    bottom_right_point = [(1+trap_bottom_width)*image_width/2, image_height]
    top_left_point = [(1-trap_top_width)*image_width/2, (1-trap_height)*image_height]
    top_right_point = [(1+trap_top_width)*image_width/2, (1-trap_height)*image_height]

    vertices = np.array([bottom_left_point, bottom_right_point, top_right_point, top_left_point], np.int32)
    vertices.reshape((-1, 1, 2))

    return [vertices]



def region_of_interest(image, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon formed from `vertices`.
    The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(image)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
    else:
            ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_lines(image, rho, theta, threshold, min_line_length, max_line_gap):
    """
    applies the probabilistic Hough transform, which identifies the straight lines on an image
    :param image: the output of a Canny transform.
    :param rho: distance resolution in pixels of the Hough grid
    :param theta: angular resolution in radians of the Hough grid
    :param threshold: minimum number of votes (intersections in Hough grid cell)
    :param min_line_length: minimum number of pixels making up a line
    :param max_line_gap: maximum gap in pixels between connectable line segments
    :return: an image with Hough lines drawn on it      #######?
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)


def get_line_equations(image, lines, slope_threshold):
    """
    :param image:
    :param lines: set of lines as returned by hough_lines
    :param slope_threshold: threshold for line selection
    :return: the equations of both lines, of the form y=mx+b
    """
    """
    1) separating line segments by their slope ((y2-y1)/(x2-x1)) to decide if they are part of the left
    line vs. the right line. 
    # carefull, harsh asumptions made
    2) average the position of each of the lines and extrapolate to the top and bottom of the lane.
    """
    # In case of error, don't draw the line(s)
    if lines is None:
            return
    if len(lines) == 0:
            return
    draw_right = True
    draw_left = True

    # Find slopes of all lines
    # But only care about lines where abs(slope) > slope_threshold
    slopes = []
    new_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

        # Calculate slope
        if x2 - x1 == 0.:  # corner case, avoiding division by 0
                slope = 999.  # practically infinite slope
        else:
                slope = (y2 - y1) / (x2 - x1)

        # Filter lines based on slope
        if abs(slope) > slope_threshold:
                slopes.append(slope)
                new_lines.append(line)

    lines = new_lines

    # Split lines into right_lines and left_lines, representing the right and left lane lines
    # Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
    right_lines = []
    left_lines = []
    for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            img_x_center = image.shape[1] / 2  # x coordinate of center of image
            if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
                    right_lines.append(line)
            elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
                    left_lines.append(line)

    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []

    for line in right_lines:
            x1, y1, x2, y2 = line[0]

            right_lines_x.append(x1)
            right_lines_x.append(x2)

            right_lines_y.append(y1)
            right_lines_y.append(y2)

    if len(right_lines_x) > 0:
            right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    else:
            right_m, right_b = 1, 1
            draw_right = False

    # Left lane lines
    left_lines_x = []
    left_lines_y = []

    for line in left_lines:
            x1, y1, x2, y2 = line[0]

            left_lines_x.append(x1)
            left_lines_x.append(x2)

            left_lines_y.append(y1)
            left_lines_y.append(y2)

    if len(left_lines_x) > 0:
            left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
    else:
            left_m, left_b = 1, 1
            draw_left = False

    return right_m, right_b, left_m, left_b



# End helper functions


# Main script



def lane_detector(frame):
    color_filtered_image = filter_colors(frame)
    gray = grayscale(color_filtered_image)
    vertices = get_ROI_vertices(gray)
    ROI_image = region_of_interest(gray, vertices)
    Canny = canny(ROI_image, low_threshold, high_threshold)
    Hough_lines = hough_lines(Canny, rho, theta, threshold, min_line_length, max_line_gap)
    right_m, right_b, left_m, left_b = get_line_equations(frame, Hough_lines, slope_threshold)
    return right_m, right_b, left_m, left_b
    
