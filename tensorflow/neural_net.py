#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:07:55 2019

@author: samuel
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
hello = tf.constant("hello")
session = tf.Session()



mnist = tf.keras.datasets.mnist



# loading data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()



#normalizing feature data to be between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()

# takes 28x28 image and flatterns it 
model.add(tf.keras.layers.Flatten())

#sets 128 neurons in second and third layer and set activation function rectify linear 
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# set final layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.softmax))

# Set model parameters
model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


model.fit(x_train, y_train, epochs=3)


value_loss, value_acc = model.evaluate(x_test,y_test)

model.save('num_model.model')

new_prediction_model = tf.keras.models.load_model('num_model.model')

prediction = new_prediction_model.predict(x_test)


print('First ten numbers in test set' + '\n'*5)

for i in range(10):
    plt.imshow(x_test[i])
    plt.show()
    plt.clf()
    print('Model predicted: ')
    print(np.argmax(prediction[i]))
    















