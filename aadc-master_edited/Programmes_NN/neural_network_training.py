"""trains the neural network"""

from neural_network import *
from database_creator import *


input = load_input_with_shuffle(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_entrainement")
# print(input.shape[0])
input_matrix = input[:(input.shape[0]-1),:]
experimental_input = input[input.shape[0]-1::input.shape[0]]
# print(input_matrix.shape)
# print(experimental_input)
m = input_matrix.shape
NN1 = Nnetwork([m[0],30,30,30,30,30,1],[sigmoid,sigmoid,sigmoid,sigmoid,sigmoid,sigmoid])



def trains_neural(input_matrix,experimental_matrix,NN):
    NN.epoch(input_matrix,experimental_matrix)



trains_neural(input_matrix,experimental_input,NN1)


test_images = load_input_with_shuffle(r"C:\Users\Antonio\Documents\Projet_Audi_Cup\Reconnaissance_STOP\panneaux_traites_test")
# print(input.shape[0])
test_matrix = input[:(input.shape[0]-1),:]

test_output = input[input.shape[0]-1::input.shape[0]]



def test_accuracy(NN,test_images,result):
    A = NN.forward_propagation(test_images,result)[0]
    # print(A)
    score = 0
    for i in range(A.shape[1]):
        x = result[0,i]

        if A[0,i]>=0.5:
            y = 1
        else:
            y = 0
        if x == y:
            score+=1
        print(A[0,i])
        print(x,y)
    return score/A.shape[1]*100

print(test_accuracy(NN1,test_matrix,test_output))

