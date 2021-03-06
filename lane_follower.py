'''
1) find the intersection point of the two lines (given in lane_detector.py by get_line_equations(),
   which gives the direction of the lane
2) deduce the order (an angle here) to give to the hardware. Note that it also requires a speed, to be computed after:
    for the angular speed, we will probably choose to output something proportional to the difference on angles,
    with a coefficient chosen so that the driving is smooth.
'''

import numpy as np
import lane_detector as ld


def vanishingPoint(right_m, right_b, left_m, left_b):
    """
    each lane is a line defined by an equation of the form y = side_m*x + side_b
    :param right_m: slope of the right lane
    :param right_b: y-intercept of the right lane
    :param left_m: slope of the left lane
    :param left_b: y-intercept of the left lane
    :return: the intersect of the two lines(i.e. the point the car should head for)
    """
    if right_m==left_m: #parralel lines
        return
    else:
        x = (right_m-left_m)/(left_b-right_b)
        y = right_m*x + right_b
        return x,y


def direction(image, vanishing_point):
    """
    :param image: the reference image (to get its shape. We can also get it once and put it as a global parameter as long as the camera doesn't change.
    :param vanishingPoint: the point to which the car should head (coordinates in relative size of the image)
    :return: an angle in radians, positive when it should turn right
    """
    if vanishingPoint is None:
        return
    else:
        x_center = image.shape[1]/2 #x coordinate of the center of the image
        x,y = vanishing_point
        theta = np.arctan( (x-x_center)/y )
        return theta


def lane_follower(frame):
    right_m, right_b, left_m, left_b = ld.lane_detector(frame)
    vanishing_point = vanishingPoint(right_m, right_b, left_m, left_b)
    theta = direction(frame, vanishing_point)
    return theta
