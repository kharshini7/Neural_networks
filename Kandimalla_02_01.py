# Kandimalla, Harshini
# 1001_960_046
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf

def mse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()

def SVM_loss(y,true_class_index,delta=1):
    # print("True class: ",true_class_index)
    margins = np.maximum(0, y - y[true_class_index] + delta)
    margins[true_class_index] = 0
    loss_i = np.sum(margins)
    return loss_i

def valid(X_train,Y_train,activations,weights,loss):
    input_arr= X_train
    activation_vals=[]
    layer_count=0
    for activation in activations:
      net=np.dot(input_arr,weights[layer_count])
      if activation.lower()=="linear":
        activation_vals.append(net)
      elif activation.lower()=="sigmoid":
        activation_vals.append(1.0 / (1 + np.exp(-net)))
      elif activation.lower()=="relu":
        net[net<0]=0
        activation_vals.append(net)
      input_arr=activation_vals[layer_count]
      layer_count+=1
    if loss=="svm":
      pass
    if loss=="mse":
      error=mse(Y_train,activation_vals[-1])
    if loss=="cross_entropy":
      error=tf.nn.softmax_cross_entropy_with_logits()
    return error

def valid_predict(X_train,Y_train,activations,weights,loss):
    input_arr= X_train
    # print("shapppee",X_train.shape)
    activation_vals=[]
    layer_count=0
    for activation in activations:
      net=np.dot(input_arr,weights[layer_count])
      if activation.lower()=="linear":
        activation_vals.append(net)
      elif activation.lower()=="sigmoid":
        activation_vals.append(1.0 / (1 + np.exp(-net)))
      elif activation.lower()=="relu":
        net[net<0]=0
        activation_vals.append(net)
      input_arr=activation_vals[layer_count]
      layer_count+=1
    return activation_vals[-1]
def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],weights=None,seed=2):
    input_dimension=X_train.shape[1]
    bias=[]
    flag=1
    if weights==None:
      # flag=0
      weights=[]
      for i in range(len(layers)):
        np.random.seed(seed)    
        output_dimension=layers[i] 
        weights.append(np.random.randn(input_dimension, output_dimension))
        # print("output_dimension",output_dimension)
        # print("input_dimension",input_dimension)
        bias.append(np.random.randn(1,output_dimension))
        # print("weights.shape",weights[i].shape)
        input_dimension=output_dimension
    else:
      layer_count=0
      for activation in activations:
            bias.append([weights[layer_count][-1]])
            print("biasss",bias)
            weights[layer_count]=weights[layer_count][:-1]
            layer_count+=1
      print(type(weights))

    start=int(np.floor(validation_split[0]*X_train.shape[0]))
    end=int(np.floor(validation_split[1]*X_train.shape[0]))
    X_train_split=X_train[:start]
    Y_train_split=Y_train[:start]
    X_validation=X_train[start:end]
    Y_validation=Y_train[start:end]
    list_of_errors = []
    for i in range(epochs):
      for j in range(0,X_train.shape[0],batch_size):
          # print("hi")
          errorval=0
          layer_count=0
          activation_vals=[]
          input_arr= X_train[j:j+batch_size]
          for activation in activations:
            # print("hi",input_arr.shape)
            net=np.dot(input_arr,weights[layer_count])
            if activation.lower()=="linear":
              activation_vals.append(net)
            elif activation.lower()=="sigmoid":
              activation_vals.append(1.0 / (1 + np.exp(-net)))
            elif activation.lower()=="relu":
              net[net<0]=0
              activation_vals.append(net)
            input_arr=activation_vals[layer_count]
            layer_count+=1
          if loss=="svm":
            error=SVM_loss(Y_train[j:j+batch_size],)
          if loss=="mse":
            error=mse(Y_train[j:j+batch_size],activation_vals[-1])
          if loss=="cross_entropy":
            error=tf.nn.softmax_cross_entropy_with_logits(Y_train[j:j+batch_size],activation_vals[-1])
          print("error",error)
          weights_error= alpha*error
          # print("weights before",weights)
          weights=weights-weights_error
          # print("weights after",weights)
          # print("bias",bias)
          # print("error",error)
          # print("weights",type(weights))
  
      list_of_errors.append(valid(X_validation,Y_validation,activations,weights,loss))
    valid_predictions=valid_predict(X_validation,Y_validation,activations,weights,loss)
    for k in range(len(weights)):
      # print(weights[k].shape)
      # print(type(weights[k]))
      # print(bias[k].shape)
      bias[k]=np.asarray(bias[k])
      # print(type(bias[k].shape))
      weights[k]=np.float32(np.hstack((weights[k].T,bias[k].T))).T
      
    # print(weights)
    print(Y_validation.shape)
    print(valid_predictions.shape)
    print(list_of_errors)
    return weights,list_of_errors,valid_predictions
