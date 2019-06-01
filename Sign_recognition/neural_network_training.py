"""trains the neural network"""

from neural_network import *
from input_creator import *

CATEGORY_NUMBER = 3

input = load_input_with_shuffle(
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_entrainement",
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_entrainement\attention46.jpg",
    CATEGORY_NUMBER)

input_matrix1 = input[:(input.shape[0] - CATEGORY_NUMBER), :]
experimental_input1 = input[(input.shape[0] - CATEGORY_NUMBER):input.shape[0], :]
inp, outp = input_matrix1.shape, experimental_input1.shape
NN1 = Nnetwork([inp[0], 5, 5, outp[0]], [sigmoid, sigmoid, sigmoid, sigmoid])

dev_images = load_input_with_shuffle(
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_dev",
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_dev\attention0.jpg",
    CATEGORY_NUMBER)

dev_inp = dev_images[:(input.shape[0] - CATEGORY_NUMBER), :]
dev_out = dev_images[(input.shape[0] - CATEGORY_NUMBER):input.shape[0], :]


def trains_neural(input_matrix, experimental_matrix, NN, dev_input, dev_output):
    NN.epoch_with_dev_set(input_matrix, experimental_matrix, dev_input, dev_output)


trains_neural(input_matrix1, experimental_input1, NN1, dev_inp, dev_out)

test_images = load_input_with_shuffle(
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_test",
    r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle\trianglesred_test\attention23.jpg",
    CATEGORY_NUMBER)

test_inp = test_images[:(input.shape[0] - CATEGORY_NUMBER), :]
test_out = test_images[(input.shape[0] - CATEGORY_NUMBER):input.shape[0], :]

print("On the test set the NN has an accuracy of: " + str(NN1.accuracy(test_inp, test_out)))

NN1.save(r'C:\Users\Antonio\Documents\Projet_Autonomous_Driving\aadc\Sign_recognition\Neural_networks', "red_triangles")
