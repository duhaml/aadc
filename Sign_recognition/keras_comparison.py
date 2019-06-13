"""This code is not clean it is used to compare on a graph the accuracy of the Keras NN and the NN from scratch"""

from neural_network_training import *
import keras
import sklearn
import matplotlib
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

(X_mat_train, Y_train), (X_mat_test, Y_test) = mnist.load_data()
n_samples_train = X_mat_train.shape[0]
n_samples_test = X_mat_test.shape[0]
n = X_mat_train.shape[1]

print(X_mat_train.shape, Y_train.shape)

d = n * n
X_train = X_mat_train.reshape(n_samples_train, d) / 255.
X_test = X_mat_test.reshape(n_samples_test, d) / 255.

print(X_train.shape, X_test.shape)


def create_output(output, categories):
    new_output = np.zeros((output.shape[0], categories))
    for i in range(output.shape[0]):
        new_output[i, output[i]] = 1
    return new_output


Y_train = create_output(Y_train, 10)
Y_test = create_output(Y_test, 10)
print(Y_train[:5, :])

X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
X_train, X_reste, Y_train, Y_reste = train_test_split(X_test, Y_test, test_size=0.8, random_state=0)

model = Sequential()
model.add(Dense(128, activation="sigmoid", input_shape=(784,)))
model.add(Dense(10, activation="sigmoid", name="representation"))
model.compile(loss="categorical_crossentropy", optimizer='sgd')


def accuracy(predictions, values):
    score = 0
    # print(output.shape)

    # d = {"out1": 0, "out2": 0, "out3": 0, "A1": 0, "A2": 0, "A3": 0}
    for i in range(predictions.shape[0]):
        argmaxout, argmaxA = 0, 0
        maxout, maxA = 0, 0
        for j in range(10):
            if values[i, j] > maxout:
                maxout = values[i, j]
                argmaxout = j

            if predictions[i, j] > maxA:
                argmaxA = j
                maxA = predictions[i, j]
        # print(argmaxout,argmaxA)
        # d["out"+str(argmaxout + 1)]+=1
        # d["A"+str(argmaxA + 1)]+=1
        if argmaxA == argmaxout:
            score += 1
    # print(d)
    return score / predictions.shape[0] * 100


accuracy1, accuracy2 = [0], [0]

accuracy2 = NN1.epoch1(X_train.T, Y_train.T, X_val.T, Y_val.T, epoch_number2=1000)

for i in range(1, 1001):
    if i % 50 == 0:
        print(i)
        results = model.fit(X_train, Y_train, batch_size=1000, epochs=i,
                            verbose=0, validation_data=(X_val, Y_val))
        scores = model.predict(X_val, verbose=0)
        accuracy1.append(accuracy(scores, Y_val))
        abs.append(i)

print(accuracy1, accuracy2)

plt.plot(abs, accuracy1, color="blue", label='Keras')
plt.plot(abs, accuracy2, color="red", label='from scratch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy on validation test (%)')
plt.legend()
plt.show()

# Results

# [0, 9.3, 12.72, 23.66, 37.88, 63.38, 74.52, 78.14, 80.74, 82.52000000000001, 83.82, 85.04, 85.64, 86.28, 86.8, 86.96000000000001, 87.36, 87.62, 87.68, 87.78, 87.92]
# [0, 10.52, 10.52, 10.54, 18.2, 32.440000000000005, 38.92, 40.06, 45.92, 44.42, 45.06, 47.06, 49.16, 50.62, 51.76, 52.459999999999994, 53.080000000000005, 53.22, 53.779999999999994, 54.059999999999995, 54.339999999999996]
