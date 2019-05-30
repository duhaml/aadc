"""functions used to create the input matrix of the simple neural network"""


import os
from PIL import Image
import numpy
import cv2
# from image_resizer import *


help_components = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 100, 100]]]}
scale_g = 0.5

def image2array(img):
    """transforms image in ppm format to array"""
    img = Image.open(img)
    return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0], 3)


def show_image(img, title, scale=scale_g):
    """takes an image and show it with the right window size"""
    newx, newy = int(img.shape[1] * scale), int(img.shape[0] * scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (newx, newy))
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_image_component(image, components, show=False):
    """takes an image and shows the mask of each components
    on components list
    The component list has lists of 2 lists with 3 elements """

    # convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # resulting list
    res = []
    # use of the threshold technique to select the mask for each component
    for component in components.keys():
        s = hsv.shape
        thresh = components[component]
        mask = numpy.zeros((s[0], s[1]))
        for threshold in thresh:
            lower_comp = numpy.array(threshold[0])
            higher_comp = numpy.array(threshold[1])
            mask += cv2.inRange(hsv, lower_comp, higher_comp)
        if show:
            show_image(mask, component)
        res.append(mask)
    return res


def reshape_masks(masks, category_number):
    """transforms the list of image_masks into a vector to vector"""
    final_mask = numpy.zeros((category_number,1))
    for mask in masks:
        final_mask = numpy.concatenate((numpy.reshape(mask.T, [mask.shape[0] * mask.shape[1], 1]),final_mask),axis = 0)

    return final_mask


def normalisation(image_path,components,category_number):
    """takes the image path and returns the vector with values between 0 and 1
    for the masks of each component"""
    matrix_img = image2array(image_path)
    masks = detect_image_component(matrix_img, help_components)
    vector = reshape_masks(masks,category_number)
    normalized_vector = vector / 255
    return normalized_vector

