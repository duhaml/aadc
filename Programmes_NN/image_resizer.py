"""file used to select good images and transfer to a file to treat and normalize them
to have a good database."""

from PIL import Image
import os
from shutil import copyfile
import random as rd

"""CONSTANTS"""
basewidth = 45


def resize(image, basewidth):
    """return the resized basewidth*basewidth image"""
    img = Image.open(image)
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    img.save(image)
    return (img.size)


def resize_file(image_directory):
    """resizes the images to have the bandwidth"""
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            resize(image_directory + '\\' + filename, basewidth)


#resize_file(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites")


def directory_renamer(directory):
    "renames all the file in a directory so that they have the prefix"
    i = 0
    for root, dirs, files in os.walk(directory):

        for filename in files:
            os.rename(directory + '\\' + filename, directory + '\\' + str(i) + ".ppm")
            i += 1


# directory_renamer(r"D:\GTSRB\Training\signs")


def digit_number(n):
    """donne les zeros a mettre on fonction du nombre de chiffre un ou deux"""
    if n > 9:
        return "000"
    else:
        return "0000"


def create_directory(path, new_directory):
    """transfers all the data in the small files to a big file"""
    for i in range(43):

        directory = path + '\\' + digit_number(i) + str(i)

        for root, dirs, files in os.walk(directory):
            for filename in files:
                copyfile(directory + '\\' + filename, new_directory + '\\' + str(i) + "_" + filename)


# create_directory(r"D:\GTSRB\Training",r"D:\GTSRB\Training\signs")


def expand_directory(path, new_directory, file_number, file_type):
    """copy file_number random files from path to new_directory to directory"""
    already_reached = []
    i = 0
    while i < file_number:

        x = rd.randint(0, 240)

        if x not in already_reached:
            copyfile(path + '\\' + str(x) + ".ppm", new_directory + '\\' + file_type + str(x) + ".ppm")

        already_reached.append(x)
        i += 1

# expand_directory(r"D:\GTSRB\Training\signs",r"D:\GTSRB\Training\not_stop",540)


def file_remover(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if "GT" in filename:
                os.remove(directory + "\\" + filename)
