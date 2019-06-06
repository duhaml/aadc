import sys

sys.path.insert(0, r'C:\Users\Antonio\Documents\Projet_Autonomous_Driving\aadc\Detection_panneaux')

import Detection_module as dm
from neural_network import *
from input_creator import *
import cv2
import numpy as np

# CONSTANTS
BASEWIDTH = 28
HELP_COMPONENTS = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 100, 100]]]}
SCALE_G = 0.5
POLYGON_CATEGORIES = {"triangles": {"red": ["attention", "priorite", "autre"], "blue": []},
                      "rectangles": {"red": [], "blue": ["parking", "passage_p", "autre"]},
                      "circles": {"red": ["limvit20", "limvit30", "stop", "autre"],
                                  "blue": ["fleche_d", "fleche_g", "autre"]}}
TEXT_H, TEXT_W = 17, 70


def show_image(img, title, scale=SCALE_G):
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
        mask = np.zeros((s[0], s[1]))
        for threshold in thresh:
            lower_comp = np.array(threshold[0])
            higher_comp = np.array(threshold[1])
            mask += cv2.inRange(hsv, lower_comp, higher_comp)
        if show:
            show_image(mask, component)
        res.append(mask)
    return res


def resize(image, basewidth=BASEWIDTH):
    """takes the """
    return cv2.resize(image, dsize=(basewidth, basewidth), interpolation=cv2.INTER_CUBIC)


def reshape_masks(masks, category_number):
    """transforms the list of image_masks into a vector to vector"""
    final_mask = np.reshape(masks[0].T, [masks[0].shape[0] * masks[0].shape[1], 1])
    for i in range(1, len(masks)):
        mask = masks[i]
        final_mask = np.concatenate((np.reshape(mask.T, [mask.shape[0] * mask.shape[1], 1]), final_mask), axis=0)

    return final_mask


def normalisation(matrix_img, category_number, components=HELP_COMPONENTS):
    """takes the image path and returns the vector with values between 0 and 1
    for the masks of each component"""
    masks = detect_image_component(matrix_img, components)
    vector = reshape_masks(masks, category_number)
    normalized_vector = vector / 255
    return normalized_vector


def draw_object(image, object_name, location, category):
    """takes an image, the object name to be shown, the location (x,y, height, width)
    and drawing a square around the object"""
    (x, y, h, w) = location
    if category == "trianglesred":
        color = (255, 0, 0)
    elif category == "rectanglesred":
        color = (0, 255, 0)
    elif category == "circlesred":
        color = (0, 0, 255)
    elif category == "circlesblue":
        color = (0, 255, 255)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.fillConvexPoly(image, np.array(
        [[[x - 2, y - TEXT_H]], [[x - 2 + TEXT_W, y - TEXT_H]], [[x - 2 + TEXT_W, y]], [[x - 2, y]]]), color=color)
    cv2.putText(image, object_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


# img = np.zeros((1000,1000))
# draw_object(img,"stop",(120,120,30,30),"circlesred")
# cv2.imshow("test",img)
# cv2.waitKey()
# cv2.destroyAllWindows()

def detected_signs(image, nns, show=False):
    """takes an image matrix and a dictionnary of neural networks and
    returns the image with the located signs"""

    classed_polygons = dm.easy_give_signs(image)
    # print(classed_polygons)
    output_categories = {"triangles": {"red": [], "blue": []},
                         "rectangles": {"red": [], "blue": []},
                         "circles": {"red": [], "blue": []}}
    for polygon_cat in classed_polygons.keys():
        for color in classed_polygons[polygon_cat].keys():
            polygons = classed_polygons[polygon_cat][color]
            for polygon in polygons:
                location, img_polygon = polygon
                # find the category
                resized_img = resize(img_polygon, basewidth=BASEWIDTH)
                neural_net = nns[polygon_cat + color]
                category_nbre = neural_net.layers[-1]
                input_vect = normalisation(resized_img, category_nbre)
                output_vect = neural_net.calculate(input_vect)
                argmaxi = np.argmax(output_vect)
                # for future functions
                category = POLYGON_CATEGORIES[polygon_cat][color][argmaxi]
                output_categories[polygon_cat][color].append(category)

                # draw the rectangle around the object
                if category != "autre":
                    # print(polygon)
                    print(category, polygon_cat + color)
                    draw_object(image, category, location, polygon_cat + color)

    if show:
        show_image(image, "recognized_signs")

    return output_categories
