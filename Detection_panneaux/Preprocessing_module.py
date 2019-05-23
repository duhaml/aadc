import cv2
import numpy as np

# Image size from rasppi camera


# CONSTANTS
components = {"blue": [[[95, 70, 50], [166, 255, 255]]],
              "red": [[[0, 50, 50], [14, 255, 255]], [[160, 100, 100], [179, 255, 255]]],
              "white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 50, 60]]]}
scale_g = 0.5


def show_image(img, title, scale=scale_g):
    """takes an image and show it with the right window size"""
    newx, newy = int(img.shape[1] * scale), int(img.shape[0] * scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, (newx, newy))
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_image_component(image, components, scale=scale_g, show=False):
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
        mask = np.zeros((s[0], s[1]))
        for threshold in thresh:
            lower_comp = np.array(threshold[0])
            higher_comp = np.array(threshold[1])
            mask += cv2.inRange(hsv, lower_comp, higher_comp)
        if show:
            show_image(mask, component)
        res.append(mask)
    return res


def blur_image(image, show=False):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    if show:
        show_image(blur, 'blur')
    return blur


def image_contour(image, show=False):
    """takes an image filename and shows the smoothed image
    using a Gaussian Filter and the Canny edge detection technique"""

    # application of the Gaussian filter
    blur = blur_image(image)
    # detection of the edges
    blur_copy = np.uint8(blur)
    edges = cv2.Canny(blur_copy, 100, 200)
    if show:
        show_image(edges, 'edges')
    return edges


def each_image(image_list, function, shows=False, i=0):
    treated_images = []
    for image in image_list:
        treated_images.append(function(image, show=shows))
    return treated_images


"""Tests"""
# show_image(cv2.imread(r'Test_images\test_rouge_2.jpg'),'normal')
# detect_image_component(cv2.imread(r'Test_images\test_rouge.jpg'),components, show = True)
# blur_image(detect_image_component(cv2.imread(r'Test_images\test_rouge.jpg'),components)[0], show = True)
# image_contour(detect_image_component(cv2.imread(r'Test_images\test_rouge.jpg'),components)[1], show = True)
# each_image(detect_image_component(cv2.imread(r'Test_images\test_rouge.jpg'),components), blur_image, shows = True)
