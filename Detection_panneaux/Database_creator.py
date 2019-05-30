"""this file creates a database of images from the detection module"""

from PIL import Image
import os
import cv2
from shutil import copyfile
import random as rd
import Detection_module as dm

BASEWIDTH = 28

def resize(image, basewidth = BASEWIDTH):
    """takes the """
    return cv2.resize(image, dsize=(basewidth, basewidth), interpolation=cv2.INTER_CUBIC)

def create_directory(old_dir, new_dir, basewidth = BASEWIDTH):
    i = 0
    for root, dirs, files in os.walk(old_dir):
        for filename in files:
            # print(filename)
            classed_polygons = dm.easy_give_signs(cv2.imread(old_dir + '\\' + filename))
            for poly_type in classed_polygons.keys():
                for color in classed_polygons[poly_type].keys():
                    for image in classed_polygons[poly_type][color]:
                        if i % 100 == 0:
                            print(i)
                        resized_img = resize(image, basewidth = basewidth)
                        cv2.imwrite(new_dir + '\\' + poly_type + color + '\\' + str(i) + '.jpg',resized_img)
                        i+=1
            if i > 10000:
                break


def directory_renamer(prefix, directory):
    "renames all the file in a directory so that they have the prefix"
    i = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            os.rename(directory + '\\' + filename, directory + '\\' + prefix + str(i) + ".jpg")
            i += 1

def directory_converter(old_directory,new_directory,new_format):
    """takes the path of a directory and converts the image format to new_format"""
    i = 0
    for root, dirs, files in os.walk(old_directory):
        for filename in files:
            image = cv2.imread(old_directory + '\\' + filename)
            filename = filename.split('.')
            cv2.imwrite(new_directory + '\\' + filename[0] + new_format,image)
            i += 1

