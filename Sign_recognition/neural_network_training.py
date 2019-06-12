"""trains the neural network"""

from neural_network import *
from input_creator import *

CATEGORY_NUMBER = 3
CATEGORIES = ["attention","priorite","autre"]



def trains_neural(input_path, first_input, dev_path, first_dev):
    input = load_input_with_shuffle(input_path, first_input,CATEGORY_NUMBER,CATEGORIES)
    input_matrix = input[:(input.shape[0] - CATEGORY_NUMBER), :]
    experimental_matrix = input[(input.shape[0] - CATEGORY_NUMBER):input.shape[0], :]
    inp, outp = input_matrix.shape, experimental_matrix.shape
    NN1 = Nnetwork([inp[0], 6, outp[0]], [sigmoid, sigmoid, sigmoid, sigmoid])


    dev_images = load_input_with_shuffle(dev_path,first_dev,CATEGORY_NUMBER,CATEGORIES)

    dev_input = dev_images[:(input.shape[0] - CATEGORY_NUMBER), :]
    dev_output = dev_images[(input.shape[0] - CATEGORY_NUMBER):input.shape[0], :]

    NN1.epoch_with_dev_set(input_matrix, experimental_matrix, dev_input, dev_output)
    return NN1

NNet = trains_neural(r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_entrainement",
                      r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_entrainement\attention46.jpg",
                      r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_dev",
                      r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_dev\attention0.jpg")



def test_nn(test_path, first_test,NN1):
    test_images = load_input_with_shuffle(
        r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_test",
        r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_test\attention23.jpg",
        CATEGORY_NUMBER,CATEGORIES)
    test_inp = test_images[:(test_images.shape[0] - CATEGORY_NUMBER), :]
    test_out = test_images[(test_images.shape[0] - CATEGORY_NUMBER):test_images.shape[0], :]

    return ("On the test set the NN has an accuracy of: " + str(NN1.accuracy(test_inp, test_out)))

print(test_nn(r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_test",
              r"C:\Users\Antonio\Documents\Projet_Autonomous_Driving\Dataset_global\Dataset_entrainement_triangle_rouge\trianglesred_test\attention23.jpg",
              NNet))

NNet.save(r'C:\Users\Antonio\Documents\Projet_Autonomous_Driving\aadc\Sign_recognition\Neural_networks', "red_triangles")
