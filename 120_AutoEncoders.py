from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adadelta
from keras.utils import np_utils

# from keras.utils.visualize_util import plot
from IPython.display import SVG
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import model_to_dot

import numpy as np

from matplotlib import pyplot as plt


# Image Dimension
input_unit_size = 28*28

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# function to plot digits
def draw_digit(data, row, col, n):
    size = int(np.sqrt(data.shape[0]))
    plt.subplot(row, col, n)
    plt.imshow(data.reshape(size, size))
    plt.gray()



# Normalize
X_train = X_train.reshape(X_train.shape[0],input_unit_size)
X_train = X_train.astype('float32')
X_train /=255
print('X_train shape:', X_train.shape)


#------------------------construction Model-----------------------------------
inputs = Input(shape=(input_unit_size,))
x = Dense(144, activation='relu')(inputs)
outputs = Dense(input_unit_size)(x)
model = Model(input=inputs, output=outputs)
model.compile(loss='mse', optimizer='adadelta')

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

model.fit(X_train, X_train, nb_epoch=5, batch_size=258)



#--------------------visualize Input Layers-------------------------------------
# plot the images from input layer
show_size = 5
total = 0
plt.figure(figsize=(5,5))
for i in range(show_size):
    for j in range(show_size):
        draw_digit(X_train[total], show_size, show_size, total+1)
        total+=1
plt.show()


# ----------------------------------Visulize Hidden Layers---------------------------------------
# plot the encoded (compressed) layer image
get_layer_output = K.function([model.layers[0].input], [model.layers[1].output])

hidden_outputs = get_layer_output([X_train[0:show_size**2]])[0]

total = 0
plt.figure(figsize=(5,5))
for i in range(show_size):
    for j in range(show_size):
        draw_digit(hidden_outputs[total], show_size, show_size, total+1)
        total+=1
plt.show()


# ---------------------Visulize output Layers ----------------------------------------
# Plot the decoded (de-compressed) layer images
get_layer_output = K.function([model.layers[0].input], [model.layers[2].output])

last_outputs = get_layer_output([X_train[0:show_size**2]])[0]

total = 0
plt.figure(figsize=(5,5))
for i in range(show_size):
    for j in range(show_size):
        draw_digit(last_outputs[total], show_size, show_size, total+1)
        total+=1
plt.show()


