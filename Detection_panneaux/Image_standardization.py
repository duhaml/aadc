import os

from PIL import Image

# CONSTANTS
stand_size = (2592, 1944)
stand_width = 1500
ratio = stand_size[0] / stand_size[1]


# FORMAT CHANGE
def to_jpg(img_path):
    """saves image to jpg format"""
    img = Image.open(img_path)
    file_name, file_extension = os.path.splitext(img_path)
    img.save(file_name + '.jpg')


def jp2jpg_directory(directory, new_directory):
    """transfers jp2 images from directory to new_directory in jpg format"""
    i = 0
    print(i)
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if i % 100 == 0:
                print("Image treated: " + str(i))
            img = Image.open(directory + "\\" + filename)
            img.save(new_directory + "\image_" + str(i) + ".jpg")

            i += 1


# SIZE CHANGE
def resize(image_path, size=stand_size):
    """takes image path and saves it in the desired size by reshaping the image
    and returns the final size of the image"""
    img = Image.open(image_path)
    img = img.resize((stand_width, int(stand_width / ratio)), Image.ANTIALIAS)
    fileName, fileExtension = os.path.splitext(image_path)
    img.save(fileName + fileExtension)
    return img.size


def crop(image_path, size=stand_size):
    """takes image path and saves it in the desired size by reshaping the image
    and returns the final size of the image"""
    img = Image.open(image_path)
    width, height = img.size
    ratio = width / height
    stand_ratio = size[0] / size[1]
    if ratio > stand_ratio:
        # couper sur la longueur
        new_w = stand_ratio * height
        left = int((width - new_w) / 2)
        cropped_img = img.crop(((left, 0, width - left, height)))
    else:
        # couper sur hauteur
        new_h = width / stand_ratio
        bottom = int((height - new_h) / 2)
        cropped_img = img.crop(((0, height, width, bottom)))
    recropped_img = cropped_img.resize((size[0], size[1]), Image.ANTIALIAS)
    recropped_img.save(image_path)
    return (recropped_img.size)


def resize_directory(image_directory):
    """resizes the images to have the bandwidth"""
    for root, dirs, files in os.walk(image_directory):
        for filename in files:
            resize(image_directory + '\\' + filename)


def directory_renamer(directory):
    """renames all the file in a directory so that they have the prefix"""
    i = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            os.rename(directory + '\\' + filename, directory + '\\' + str(i) + ".ppm")
            i += 1
