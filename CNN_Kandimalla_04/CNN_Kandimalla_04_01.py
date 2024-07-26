# Kandimalla, Harshini
# 1001_960_046
# 2023_04_16
# Assignment_04_01


# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import keras
# import tensorflow.keras as keras

class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network

        """
        self.cnn_model = keras.models.Sequential()

    def add_input_layer(self, shape=(2,),name="" ):
        """
         This method adds an input layer to the neural network. If an input layer exist, then this method
         should replace it with the new input layer.
         Input layer is considered layer number 0, and it does not have any weights. Its purpose is to determine
         the shape of the input tensor and distribute it to the next layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        self.input_shape = shape
        add_input_layer = keras.layers.InputLayer(input_shape = self.input_shape, name = name)
        self.cnn_model.add(add_input_layer)

    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This method adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        #we do activation.lower() to ensure that eventhough if the mixes up the cases out program should pass right input 
        append_dense_layer = keras.layers.Dense(num_nodes, activation=activation.lower(), name=name, trainable=trainable)
        self.cnn_model.add(append_dense_layer)

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This method adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object"""
        add_conv_layer = keras.layers.Conv2D(num_of_filters, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation.lower(), name=name, trainable=trainable)
        self.cnn_model.add(add_conv_layer)
        return add_conv_layer 
    
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This method adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        add_maxpool_layer = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)
        self.cnn_model.add(add_maxpool_layer)
        return add_maxpool_layer
    
    def append_flatten_layer(self,name=""):
        """
         This method adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        self.cnn_model.add(keras.layers.Flatten(name=name))

    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This method sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        #Check if layer name given. Then nested if condition checks if the layer_number is a list of integer or single integer
        if layer_names == None:
            if type(layer_numbers) == list:
                for i in layer_numbers:
                    self.cnn_model.get_layer(index = i -1).trainable = trainable_flag
            else:
                self.cnn_model.get_layer(index = layer_numbers - 1).trainable = trainable_flag
        if not type(layer_numbers) == list:
           self.cnn_model.get_layer(layer_number=layer_numbers, layer_name=layer_names).trainable = trainable_flag
        else:
            for layer in layer_numbers:    
                self.cnn_model.get_layer(layer_number=layer, layer_name=layer_names[layer]).trainable = trainable_flag


    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This method should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """
        if layer_number == None:
            if len(self.cnn_model.get_layer(name=layer_name).get_weights()) <= 0:
                return None
            else:
                return self.cnn_model.get_layer(name=layer_name).get_weights()[0]
        else:
            if len(self.cnn_model.layers[layer_number - 1].get_weights()) <= 0 or layer_number == 0:
                return None
            elif layer_number == -1:
                return self.cnn_model.layers[layer_number].get_weights()[0]
            else:
                return self.cnn_model.layers[layer_number - 1].get_weights()[0]

    def get_biases(self,layer_number=None,layer_name=""):
        """
        This method should return the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        if layer_number != None:
            if len(self.cnn_model.get_layer(index=layer_number-1).get_weights()) <= 0 or layer_number == 0:
                return None
            elif layer_number == -1:
                return self.cnn_model.layers[layer_number].get_weights()[1]
            else:
                return self.cnn_model.layers[layer_number-1].get_weights()[1]
        else:
            if len(self.cnn_model.get_layer(name=layer_name).get_weights()) <= 0:
                return None
            else:
                return self.cnn_model.get_layer(name=layer_name).get_weights()[1]        

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This method sets the weight matrix for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        if layer_number and layer_number != 0:
            keras.backend.set_value(self.cnn_model.get_layer(index=layer_number-1).weights[0], weights)
        else:
            keras.backend.set_value(self.cnn_model.get_layer(name=layer_name).weights[0], weights) 

    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This method sets the biases for layer layer_number.
        layer numbers start from zero. Note that layer 0 is the input layer.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0. Note that layer 0 is the input layer
         and it does not have any weights or biases.
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        if layer_number and layer_number!=0:
            keras.backend.set_value(self.cnn_model.get_layer(index=layer_number - 1).weights[1], biases)
        else:
            keras.backend.set_value(self.cnn_model.get_layer(name=layer_name).weights[1], biases)

    def remove_last_layer(self):
        """
        This method removes a layer from the model.
        :return: removed layer
        """
        old_model = self.cnn_model
        self.cnn_model = keras.models.Sequential()

        for layer in old_model.layers[:-1]:
            self.cnn_model.add(layer)

        return old_model.layers[-1]

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This method loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        if model_name != "":
            self.cnn_model = keras.Sequential()
            if model_name == 'VGG16':
                self.cnn_model = tf.keras.applications.VGG16()
            if model_name == 'VGG19':
                self.cnn_model = tf.keras.applications.VGG19()
        else:
            self.cnn_model = keras.models.load_model(model_file_name)
        self.cnn_model = keras.models.Sequential(self.cnn_model.layers)
        return self.cnn_model

    def save_model(self,model_file_name=""):
        """
        This method saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        return self.cnn_model.save(model_file_name)

    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This method sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
       """
        if loss == "SparseCategoricalCrossentropy":
            self.loss = keras.losses.SparseCategoricalCrossentropy()
        elif loss == "MeanSquaredError":
            self.loss = keras.losses.MeanSquaredError()
        elif loss == "hinge":
            self.loss = keras.losses.Hinge()

    def set_metric(self,metric):
        """
        This method sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        if metric == "accuracy":
            self.metric = ['accuracy']
        else:
            self.metric = ['mse']

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        """
        This method sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        if optimizer == "SGD":
            self.optimizer = keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum)
        elif optimizer == "RMSprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate = learning_rate, momentum = momentum)
        elif optimizer == "Adagrad":
            self.optimizer = keras.optimizers.Adagrad(learning_rate = learning_rate, momentum = momentum)

    def predict(self, X):
        """
        Given array of inputs, this method calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        return self.cnn_model.predict(X.astype('float'))
    
    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this method returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        test_loss, test_accuracy = self.evaluate(X, y)
        return test_loss, test_accuracy
    
    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this method trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        self.cnn_model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metric)
        history = self.cnn_model.fit(x = X_train, y = y_train, epochs = num_epochs, batch_size = batch_size)
        return history.history['loss']
    

if __name__ == "__main__":

    cnn_hxk=CNN()
    cnn_hxk.add_input_layer(shape=(32,32,3),name="input")
    cnn_hxk.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', name="convlayer1")
    cnn_hxk.append_maxpooling2d_layer(pool_size=2, padding="same", strides=2,name="poolinglayer1")
    cnn_hxk.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="convlayer2")
    cnn_hxk.append_flatten_layer(name="flattenlayer1")
    cnn_hxk.append_dense_layer(num_nodes=10,activation="sigmoid",name="denselayer1")
    cnn_hxk.append_dense_layer(num_nodes=2,activation="relu",name="denselayer2")
    #giving only layer number as input parameter and not layer name
    weights=cnn_hxk.get_weights_without_biases(layer_number=0)
    biases=cnn_hxk.get_biases(layer_number=0)
    print("weight for layer 0",None if weights is None else weights.shape,type(weights))
    print("bias for layer 0",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_number=1)
    biases=cnn_hxk.get_biases(layer_number=1)
    print("weight for layer 1",None if weights is None else weights.shape,type(weights))
    print("bias for layer 1",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_number=2)
    biases=cnn_hxk.get_biases(layer_number=2)
    print("weight for layer 2",None if weights is None else weights.shape,type(weights))
    print("bias for layer 2",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_number=3)
    biases=cnn_hxk.get_biases(layer_number=3)
    print("weight for layer 3",None if weights is None else weights.shape,type(weights))
    print("bias for layer 3",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_number=4)
    biases=cnn_hxk.get_biases(layer_number=4)
    print("weight for layer 4",None if weights is None else weights.shape,type(weights))
    print("bias for layer 4",None if biases is None else biases.shape,type(biases))
    weights = cnn_hxk.get_weights_without_biases(layer_number=5)
    biases = cnn_hxk.get_biases(layer_number=5)
    print("weight for layer 5", None if weights is None else weights.shape, type(weights))
    print("bias for layer 5", None if biases is None else biases.shape, type(biases))

    biases=cnn_hxk.get_biases(layer_number=0)
    print("input weights for layer 0: ",None if weights is None else weights.shape,type(weights))
    print("input biases for layer 1: ",None if biases is None else biases.shape,type(biases))    
    #passing only layer name to get the weights
    weights=cnn_hxk.get_weights_without_biases(layer_name="convlayer1")
    biases=cnn_hxk.get_biases(layer_name="convlayer1")
    print("weights of convolution layer1: ",None if weights is None else weights.shape,type(weights))
    print("biases of convolution layer1: ",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_name="poolinglayer1")
    biases=cnn_hxk.get_biases(layer_name="poolinglayer1")
    print("weights of pooling layer1: ",None if weights is None else weights.shape,type(weights))
    print("weights of pooling layer1: ",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_name="convlayer2")
    biases=cnn_hxk.get_biases(layer_name="convlayer2")
    print("weights of convolution layer2: ",None if weights is None else weights.shape,type(weights))
    print("biases of convolution layer2: ",None if biases is None else biases.shape,type(biases))
    weights=cnn_hxk.get_weights_without_biases(layer_name="flattenlayer1")
    biases=cnn_hxk.get_biases(layer_name="flattenlayer1")
    print("weights of flatten layer1: ",None if weights is None else weights.shape,type(weights))
    print("biases of flatten layer1: ",None if biases is None else biases.shape,type(biases))
    weights = cnn_hxk.get_weights_without_biases(layer_name="denselayer1")
    biases = cnn_hxk.get_biases(layer_name="denselayer1")
    print("weights of dense layer1: ", None if weights is None else weights.shape, type(weights))
    print("biases of dense layer1: ", None if biases is None else biases.shape, type(biases))
    weights = cnn_hxk.get_weights_without_biases(layer_name="denselayer2")
    biases = cnn_hxk.get_biases(layer_name="denselayer2")
    print("weights of dense layer2: ", None if weights is None else weights.shape, type(weights))
    print("biases of dense layer2: ", None if biases is None else biases.shape, type(biases))
