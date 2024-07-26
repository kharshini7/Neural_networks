# Kandimalla, Harshini
# 1001_960_046
# 2023_04_02
# Assignment_03_01

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Softmax, Dropout
from keras.regularizers import L2
import unittest.mock

def confusion_matrix(y_true, y_pred, n_classes=10):
    # Compute the confusion matrix for a set of predictions
    
    # Initialize the confusion matrix to all zeros
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)
    
    # Loop over all examples in the prediction set
    for i in range(len(y_true)):
   # Get the index of the predicted class
        conf_mat[y_true[i]][y_pred[i]] += 1   # Increment the count in the confusion matrix
    # print(conf_mat)
    plt.matshow(conf_mat)
    plt.savefig('confusion_matrix.png')
    # plt.show()
    return conf_mat


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    tf.keras.utils.set_random_seed(5368) # do not remove this line
    model = Sequential()
    model.add(Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu',input_shape=(28,28,1),kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Dense(10, activation='linear',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(Activation('softmax'))
    # model.add(Softmax(axis=1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    history=model.fit(X_train, Y_train, epochs=epochs,batch_size=batch_size,validation_split=0.2)
    output=model.predict(X_test)
    # print(output)
    Y_test=np.argmax(Y_test,axis=1)
    # for i in [21,36,41,42,43,44,53,61,73,75,82,86]:
    #   print(output[i])
    output=np.argmax(output,axis=1)
    cm=confusion_matrix(Y_test,output)
    model.save('model.h5')
     
    return model,history,cm,output