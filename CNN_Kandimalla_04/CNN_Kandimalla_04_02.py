# Kandimalla, Harshini
# 1001_960_046
# 2023_04_16
# Assignment_04_02

from Kandimalla_04_01 import CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images=train_images[:200]
train_labels=train_labels[:200]
test_images=test_images[:200]
test_labels=test_labels[:200]
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'truck']

def test_for_train():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation ='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='linear'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(10))
    model.summary()
    model.compile(optimizer='RMSprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    test_train_accuracy = model.evaluate(test_images, test_labels)[1]

    cnn_hxk = CNN()
    cnn_hxk.add_input_layer(shape=(32, 32, 3), name='input_layer1')
    cnn_hxk.append_conv2d_layer(num_of_filters=32, kernel_size=(3, 3), activation='relu', name='conv2D_layer1')
    cnn_hxk.append_maxpooling2d_layer(pool_size=2, strides=2, name='pool_layer1')
    cnn_hxk.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='linear', name='conv2d_layer2')
    cnn_hxk.append_maxpooling2d_layer(pool_size=2, strides=2, name='pool_layer2')
    cnn_hxk.append_conv2d_layer(num_of_filters=64, kernel_size=(3, 3), activation='relu', name='conv2D_layer3')
    cnn_hxk.append_flatten_layer(name='flat_layer1')
    cnn_hxk.append_dense_layer(num_nodes=64, activation='Sigmoid',name='dense_layer_1')
    cnn_hxk.append_dense_layer(num_nodes=10, activation='softmax',name='dense_layer_2')
    cnn_hxk.set_loss_function('MeanSquaredError')
    cnn_hxk.set_optimizer('RMSprop')
    cnn_hxk.set_metric(['accuracy','mse'])
    cnn_hxk.train(train_images, train_labels, 5, 5)
    test_accuracy = model.evaluate(test_images, test_labels)[1]
    assert (test_train_accuracy == test_accuracy)


def test_for_evaluate():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='linear'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='sigmoid'))
    model.add(layers.Dense(10))
    model.summary()
    model.compile(optimizer='SGD', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    test_evaluate_loss, test_evaluate_accuracy = model.evaluate(test_images, test_labels)

    cnn_hxk = CNN()
    cnn_hxk.add_input_layer(shape=(32,32,3), name='input_layer1')
    cnn_hxk.append_conv2d_layer(num_of_filters= 32, kernel_size=(3,3), activation='relu', name='conv2D_layer1')
    cnn_hxk.append_maxpooling2d_layer(pool_size=2, strides=2, name='pool_layer1')
    cnn_hxk.append_conv2d_layer(num_of_filters=64, kernel_size=(3,3), activation='linear', name='conv2d_layer2')
    cnn_hxk.append_maxpooling2d_layer(pool_size=2, strides=2, name='pool_layer2')
    cnn_hxk.append_conv2d_layer(num_of_filters=64, kernel_size=(3,3), activation='relu', name='conv2D_layer3')
    cnn_hxk.append_flatten_layer(name='flat_layer1')
    cnn_hxk.append_dense_layer(num_nodes=64, activation='sigmoid',name='dense_layer_1')
    cnn_hxk.append_dense_layer(num_nodes=10, activation='softmax',name='dense_layer_2')
    cnn_hxk.set_loss_function('SparseCategoricalCrossentropy')
    cnn_hxk.set_optimizer('SGD')
    cnn_hxk.set_metric(['accuracy'])
    cnn_hxk.train(train_images, train_labels, 5, 5)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    assert(test_evaluate_accuracy<=test_accuracy)
    assert(test_evaluate_loss>=test_loss)