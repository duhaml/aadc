import os
import Preprocessing_module as pm
import math
import numpy as np
import cv2

# CONSTANTS
# create function to have better components
detection_components = {"blue": [[[95, 70, 50], [166, 255, 255]]],
                        "red": [[[0, 50, 50], [14, 255, 255]], [[160, 100, 100], [179, 255, 255]]]}
help_components = {"white": [[[0, 0, 100], [255, 100, 255]]], "black": [[[0, 0, 0], [255, 80, 60]]]}


def capt_rectangle(arr):
    """takes an array and returns the points of the biggest rectangle containing the array"""
    abs_gauche = math.inf
    abs_droit = 0
    ord_bas = math.inf
    ord_haut = 0
    n = arr.shape[0]
    for i in range(n):
        point = arr[i, 0]
        absi, ordin = point[0], point[1]
        if abs_gauche > absi:
            abs_gauche = absi
        if abs_droit < absi:
            abs_droit = absi
        if ord_bas > ordin:
            ord_bas = ordin
        if ord_haut < ordin:
            ord_haut = ordin
    origine = abs_gauche, ord_bas
    height = ord_haut - ord_bas
    width = abs_droit - abs_gauche
    return origine, height, width


def find_shape(image, show=False):
    """takes the image array and returns the arrays of the polygons in the image
    showing the polygons on the image if necessary"""
    edges = pm.image_contour(image)
    sh = image.shape
    canvas = np.zeros((sh[0], sh[1]))
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = []
    for contour in contours:
        epsilon = 0.03 * cv2.arcLength(contour, True)  # calculates the epsilon from first contour length
        polyg = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(polyg)

    polyg_img = cv2.drawContours(canvas, polygons, -1, (255, 255, 255), 1)

    if show:
        pm.show_image(polyg_img, 'polyg_imag')

    return polygons


def is_narrow(polygon):
    s = polygon.shape[0]
    maxi = 0
    mini = math.inf
    for x in range(s):
        long = cv2.arcLength(np.array([[polygon[x, 0]], [polygon[(x + 1) % s, 0]]]), False)
        if long > maxi:
            maxi = long
        if long < mini:
            mini = long
    return maxi > 2 * mini


def is_contained(polygon1, polygon2):
    """takes two polygons and sees if """
    s = polygon1.shape
    for x in range(s[0]):
        if cv2.pointPolygonTest(polygon2, (polygon1[x, 0, 0], polygon1[x, 0, 1]), False) < 0:
            return False
    return True


def principal_polygons(polygons):
    plg = polygons.copy()
    n = len(polygons)
    for i in range(n):
        for j in range(n):
            if is_contained(polygons[j], polygons[i]) and i != j:
                plg = [x for x in plg if not np.array_equal(x, polygons[j])]
    return plg


def polygone_interessant(polygons):
    """finds the interesting polygons that have a suficiently large shape
    in the list of polygons and sorts them out"""
    triangles = []
    rectangles = []
    circles = []
    polyg = []
    for polygon in polygons:
        point_nbre = len(polygon)

        if point_nbre >= 3 and cv2.isContourConvex(polygon) \
                and (cv2.contourArea(polygon) > 1200 or cv2.arcLength(polygon, True) > 500) \
                and (cv2.contourArea(polygon) < 10000 or cv2.arcLength(polygon, True) < 1500):
            # if point_nbre >= 8:
            polyg.append(polygon)
    for polygon in principal_polygons(polyg):
        point_nbre = len(polygon)
        if point_nbre > 6:
            circles.append(polygon)
        elif not (is_narrow(polygon)):
            if point_nbre == 3:
                triangles.append(polygon)
            elif point_nbre == 4:
                rectangles.append(polygon)
    return triangles, rectangles, circles


'''classification of the polygons'''


def verif_triangle(triangle, image, show=False):
    """takes the triangle and the cropped image in which it is contained
    and classes the triangle depending on the coulour of the contours inside
    the triangle"""
    white_edges = pm.image_contour(pm.detect_image_component(image, help_components, show=show)[0])
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for w_contour in white_contours:
        if is_contained(w_contour, triangle):
            black_edges = pm.image_contour(pm.detect_image_component(image, help_components, show=show)[1])
            black_contours = cv2.findContours(black_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            for b_contour in black_contours:
                if is_contained(b_contour, triangle):
                    return "rwb"
            return "rw"
    return "nothing"


def verif_rectangle(rectangle, image, show=False):
    """takes the rectangle and the cropped image in which it is contained
    and classes the rectangle depending on the coulour of the contours inside
    the rectangle"""
    white_edges = pm.image_contour(pm.detect_image_component(image, help_components, show=show)[0])
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for w_contour in white_contours:
        if is_contained(w_contour, rectangle):
            return "bw"
    return "nothing"


def verif_circle(circle, color, image, show=False):
    """takes the circle, the cropped image in which it is contained and the
    color (an integer) of the detected_sign which is 0 for blue and 1 for red
    and classes the circle depending on the colour of the contours inside
    the circle"""
    comps = pm.detect_image_component(image, help_components, show=show)
    white_edges = pm.image_contour(comps[0], show=show)
    white_contours = cv2.findContours(white_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if color:
        for w_contour in white_contours:
            if is_contained(w_contour, circle):
                return "bw"
    else:
        for w_contour in white_contours:
            if is_contained(w_contour, circle):
                black_edges = pm.image_contour(comps[1])
                black_contours = cv2.findContours(black_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                for b_contour in black_contours:
                    if is_contained(b_contour, circle):
                        return "rwb"
                return "rw"
    return "nothing"


def classify(image, triangles, rectangles, circles, circle_color):
    """takes the three kinds of signs and the circle base color
    which is 0 for blue and 1 for red and classes them """
    classed_triangles = {"rwb": [], "rw": [], "nothing": []}
    classed_rectangles = {"bw": [], "nothing": []}
    if circle_color:
        classed_circles = {"bw": [], "nothing": []}
    else:
        classed_circles = {"rwb": [], "rw": [], "nothing": []}
    for triangle in triangles:
        (x, y), h, w = capt_rectangle(triangle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_triangle = triangle - np.tile([[x, y]], (triangle.shape[0], 1, 1))
        clas = verif_triangle(translated_triangle, crop_img)
        if ratio < 2 and ratio > 0.5:
            classed_triangles[clas].append(triangle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (255, 0, 0), thickness=2)
    for rectangle in rectangles:
        (x, y), h, w = capt_rectangle(rectangle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_rectangle = rectangle - np.tile([[x, y]], (rectangle.shape[0], 1, 1))
        clas = verif_rectangle(translated_rectangle, crop_img)
        if ratio < 2 and ratio > 0.5:
            classed_rectangles[clas].append(rectangle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (0, 255, 0), thickness=2)
    for circle in circles:
        (x, y), h, w = capt_rectangle(circle)
        crop_img = image[y:y + h, x:x + w]
        ratio = h / w
        translated_circle = circle - np.tile([[x, y]], (circle.shape[0], 1, 1))
        clas = verif_circle(translated_circle, circle_color, crop_img)
        if ratio < 2 and ratio > 0.5:
            classed_circles[clas].append(circle)
            if clas != "nothing":
                cv2.polylines(image, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                              (0, 0, 255), thickness=2)
    return classed_triangles, classed_rectangles, classed_circles


def trouve_panneau(image):
    """takes the image path and
    shows the cropped image containing the traffic sign"""
    img = cv2.imread(image)
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    for polygons in masks_polygons:
        triangles, rectangles, circles = polygone_interessant(polygons)
        for triangle in triangles:
            (x, y), h, w = capt_rectangle(triangle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("triangle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        for rectangle in rectangles:
            (x, y), h, w = capt_rectangle(rectangle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("rectangle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        for circle in circles:
            (x, y), h, w = capt_rectangle(circle)
            crop_image = img[y:y + h, x:x + w]
            cv2.imshow("circle", crop_image)
            cv2.waitKey()
            cv2.destroyAllWindows()


def montre_polygones(image_path):
    """takes an images path and shows the colored polygons on the image"""
    img = cv2.imread(image_path)
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    for polygons in masks_polygons:
        triangles, rectangles, circles = polygone_interessant(polygons)
        cv2.polylines(img, triangles, True, (0, 255, 0), thickness=2)
        cv2.polylines(img, rectangles, True, (255, 0, 0), thickness=2)
        cv2.polylines(img, circles, True, (0, 0, 255), thickness=2)
    pm.show_image(img, 'polygones')


def montre_panneau_verif(image_path, title="", verify=True):
    """takes the image_path and shows the image with the highlighted
    traffic signs and by verifying that the rectangle containing the sign
    is not too flat and classes them if verify"""
    img = cv2.imread(image_path)
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    i = 0
    for polygons in masks_polygons:
        triangles, rectangles, circles = polygone_interessant(polygons)
        if verify:
            classed_triangles, classed_rectangles, classed_circles = classify(img, triangles, rectangles, circles, i)
        else:
            for triangle in triangles:
                (x, y), h, w = capt_rectangle(triangle)
                ratio = h / w
                if ratio < 2 and ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (255, 0, 0), thickness=2)
            for rectangle in rectangles:
                (x, y), h, w = capt_rectangle(rectangle)
                ratio = h / w
                if ratio < 2 and ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (0, 255, 0), thickness=2)
            for circle in circles:
                (x, y), h, w = capt_rectangle(circle)
                ratio = h / w
                if ratio < 2 and ratio > 0.5:
                    cv2.polylines(img, [np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])], True,
                                  (0, 0, 255), thickness=2)
        i += 1
    pm.show_image(img, title + 'polygones')



def give_signs(img):
    """ takes an image and returns the array of the cropped_image
    containing the polygons"""
    image_masks = pm.each_image(pm.detect_image_component(img, detection_components), pm.image_contour)
    masks_polygons = pm.each_image(image_masks, find_shape, i=1)
    i = 0
    classed_polygons = {"triangles": {"rwb": [], "rw": []},
                        "rectangles": [],
                        "circles": {"bw": [], "rwb": []}}
    for polygons in masks_polygons:
        triangles, rectangles, circles = polygone_interessant(polygons)
        """takes the three kinds of signs and the circle base color
        which is 0 for blue and 1 for red and classes them """
        for triangle in triangles:
            (x, y), h, w = capt_rectangle(triangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_triangle = triangle - np.tile([[x, y]], (triangle.shape[0], 1, 1))
            clas = verif_triangle(translated_triangle, crop_img)
            if ratio < 2 and ratio > 0.5 and clas != "nothing":
                classed_polygons["triangles"][clas].append(crop_img)
        for rectangle in rectangles:
            (x, y), h, w = capt_rectangle(rectangle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_rectangle = rectangle - np.tile([[x, y]], (rectangle.shape[0], 1, 1))
            clas = verif_rectangle(translated_rectangle, crop_img)
            if ratio < 2 and ratio > 0.5 and clas != "nothing":
                classed_polygons["rectangles"].append(crop_img)
        for circle in circles:
            (x, y), h, w = capt_rectangle(circle)
            crop_img = img[y:y + h, x:x + w]
            ratio = h / w
            translated_circle = circle - np.tile([[x, y]], (circle.shape[0], 1, 1))
            clas = verif_circle(translated_circle, i, crop_img)
            if ratio < 2 and ratio > 0.5 and clas != "nothing":
                classed_polygons["circles"][clas].append(crop_img)
        i+=1
    return classed_polygons


def show_polygones(dico):
    """test function to see if the precedent function returns the good arrays"""
    for key in dico.keys():
        if key == "rectangles":
            for rectangle in dico["rectangles"]:
                pm.show_image(rectangle,"rectangle")
        else:
            for sub_key in dico[key].keys():
                for polygon in dico[key][sub_key]:
                    pm.show_image(polygon,"circle or triangle")


def detect_directory(directory, function):
    """takes a directory path containing the images and detects the signs
    in each image in the directory"""
    for root, dirs, files in os.walk(directory):
        for filename in files:
            function(directory + '\\' + filename, filename)


"""Tests"""

"""takes the image path and shows 
the cropped images containing the traffic sign"""
# trouve_panneau(r'Test_images\test_rouge.jpg')  # marche
# trouve_panneau('test_rouge_2.jpg')
# trouve_panneau('stop.jpg') #marche
# trouve_panneau('russian_image.jpg') #marche
# trouve_panneau('signalisation.jpg') #marche
# trouve_panneau('test.jpg') #marche plus ou moins

# montre_polygones('test_rouge.jpg')
# montre_polygones('test_rouge_2.jpg')#marche
# montre_polygones('stop.jpg') #marche
# montre_polygones('russian_image.jpg') #marche
# montre_polygones('test.jpg') #marche plus ou moins
# montre_polygones('limvit.jpg')#marche

# montre_panneau_verif('limvit.jpg', verify = True)#marche
# montre_panneau_verif('russian_image.jpg')#marche
# montre_panneau_verif('test_bleu.jpg')#marche
# montre_panneau_verif('test_bleu_2_copy.ppm')
# montre_panneau_verif(r'Test_images\test_rouge.jpg')#marche
# montre_panneau_verif('test_rouge_2.jpg')#marche
# montre_panneau_verif('test_size.jpg')#ne marche pas

# print(give_signs(cv2.imread(r'Test_images\test_rouge.jpg')))

detect_directory(r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Algorithmes_panneaux\Dataset\Detection_dataset", montre_panneau_verif)
