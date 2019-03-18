"""file that is used to create a good database with normalized images
and vectors created from these images"""


import os
from PIL import Image
import numpy
from image_resizer import *

basewidth = 45

"""Transfer of the images"""
#creation of the big file of training images
# create_directory(r"D:\GTSRB\Online-Test-sort", r"D:\GTSRB\Online-Test-sort\other_signs")
#remove excel files
# file_remover(r"D:\GTSRB\Online-Test-sort\other_signs")
#renommage de dossier
# directory_renamer(r"D:\GTSRB\Online-Test-sort\STOPs")

#transfer of the data
# expand_directory(r"D:\GTSRB\Online-Test-sort\STOPs",r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_test",50,"stop_")
# #resize of the file
resize_file(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_test")




def ppm2array(img):
    """transforms image in ppm format to array"""
    img = Image.open(img)
    return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0], 3)


def reshape_img(img_array):
    """transforms image_matrix to vector"""
    return numpy.reshape(img_array.T, [img_array.shape[0] * img_array.shape[1] * 3, 1])

# print(reshape_img(ppm2array(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement\random_sign_2.ppm")).shape)
# print(ppm2array("stop2.ppm")[26,30,1])
# pour avoir la couleur (i,j,k) prendre le numéro k*basewidth*basewidth+30*j+i
# print(numpy.concatenate((numpy.array([[]]),reshape_img(ppm2array("stop2.ppm"))),axis=1))


def normalisation(image_ppm):
    """return the vector with values between 0 and 1"""
    matrix_img = ppm2array(image_ppm)
    vector = reshape_img(matrix_img)
    normalized_vector = vector / 255
    return normalized_vector

#print(normalisation(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement\random_sign_2.ppm").shape)

def load_input_with_shuffle(directory):
    """return the normalized matrix of all the directory"""
    final_matrix = normalisation(
        r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement\random_sign_2.ppm")
    i = 0
    for root, dirs, files in os.walk(directory):

        for filename in files:
            if i != 0:
                new_img = normalisation(directory + '\\' + filename)
                final_matrix = numpy.concatenate((final_matrix, new_img), axis=1)

            i += 1
    final_matrix = numpy.concatenate((final_matrix, numpy.zeros([1,final_matrix.shape[1]])), axis=0)

    compteur = 0
    for root, dirs, files in os.walk(directory):
            for filename in files:
                if "stop" in filename:
                    final_matrix[6075,compteur]=1

                compteur+=1
    final_matrix = final_matrix.T
    numpy.random.shuffle(final_matrix)
    return final_matrix.T

def load_input_without_shuffle(directory):
    """return the normalized matrix of all the directory"""
    final_matrix = normalisation(
        r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement\random_sign_2.ppm")
    i = 0
    for root, dirs, files in os.walk(directory):

        for filename in files:
            if i != 0:
                new_img = normalisation(directory + '\\' + filename)
                final_matrix = numpy.concatenate((final_matrix, new_img), axis=1)

            i += 1
    final_matrix = numpy.concatenate((final_matrix, numpy.zeros([1,final_matrix.shape[1]])), axis=0)


    return final_matrix.T




#print(load_input(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement").shape)


