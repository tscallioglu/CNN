import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops



def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples.
    mini_batches = []
    
    # Shuffles (X, Y).
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Partitions (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size).
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           
    Z1 = tf.add(tf.matmul(W1, X), b1)                     
    A1 = tf.nn.relu(Z1)                                    
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     
    A2 = tf.nn.relu(Z2)                                    
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction



def zero_pad(X, pad):     
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """

    X_pad=np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant', constant_values=(0,0))

    return X_pad



def conv_single_step(a_slice_prev, W, b):    
    """
    Applys one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z+b
    
    return Z

### FORWARD PROPOGATION
def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    # Retrieves dimensions from A_prev's shape. 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieves dimensions from W's shape.
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieves information from "hparameters".
    stride = hparameters.get("stride")
    pad = hparameters.get("pad")
    
    # Computes the dimensions of the CONV output volume.
    n_H = int((n_H_prev+2*pad-f)/stride+1)
    n_W = int((n_W_prev+2*pad-f)/stride+1)
    
    # Initializes the output volume Z with zeros.
    Z = np.zeros(((m, n_H, n_W, n_C)))
    
    # Creates A_prev_pad by padding A_prev.
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               # Loops over the batch of training examples.
        a_prev_pad = A_prev_pad[i]               # Selects ith training example's padded activation.
        for h in range(n_H):           # Loops over vertical axis of the output volume.
            # Finds the vertical start and end of the current "slice".
            vert_start = h*stride
            vert_end = h*stride+f
            
            for w in range(n_W):       # Loops over horizontal axis of the output volume.
                # Finds the horizontal start and end of the current "slice".
                horiz_start = w*stride
                horiz_end = w*stride+f
                
                for c in range(n_C):   # Loops over channels (= #filters) of the output volume.
                                        
                    # Uses the corners to define the (3D) slice of a_prev_pad.
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    # Convolves the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                    weights = W[:, :,: , c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                                        
    # Makes sure the output shape is correct.
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    # Saves information in "cache" for the backprop.
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache



def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer.
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev).
    hparameters -- python dictionary containing "f" and "stride".
    mode -- the pooling mode that would like to be used, defined as a string ("max" or "average").
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C).
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters. 
    """
    
    # Retrieves dimensions from the input shape.
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieves hyperparameters from "hparameters".
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Defines the dimensions of the output.
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initializes output matrix A.
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         # Loops over the training examples.
        for h in range(n_H):                     # Loops on the vertical axis of the output volume.
            # Finds the vertical start and end of the current "slice".
            vert_start = h*stride
            vert_end = h*stride + f
            
            for w in range(n_W):                 # Loops on the horizontal axis of the output volume.
                # Finds the vertical start and end of the current "slice".
                horiz_start = w*stride
                horiz_end = w*stride +f
                
                for c in range (n_C):            # Loops over the channels of the output volume.
                    
                    # Uses the corners to define the current slice on the ith training example of A_prev, channel c.
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    # Computes the pooling operation on the slice. 
                    # Uses an if statement to differentiate the modes. 
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice[:,:,c])

                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice[:,:,c])

    
    # Stores the input and hparameters in "cache" for pool_backward().
    cache = (A_prev, hparameters)
    
    # Making sure output shape is correct.
    assert(A.shape == (m, n_H, n_W, n_C))
    
    return A, cache


### BACKWARD PROPAGATION
def conv_backward(dZ, cache):
    """
    Implements the backward propagation for a convolution function.
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C).
    cache -- cache of values needed for the conv_backward(), output of conv_forward().
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev).
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C).
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C).
    """
    
    # Retrieves information from "cache".
    (A_prev, W, b, hparameters) = cache
    
    # Retrieves dimensions from A_prev's shape.
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieves dimensions from W's shape.
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieves information from "hparameters".
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieves dimensions from dZ's shape.
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initializes dA_prev, dW, db with the correct shapes.
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                         
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # Pads A_prev and dA_prev.
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    
    for i in range(m):                       # Loops over the training examples.
        
        # Selects ith training example from A_prev_pad and dA_prev_pad.
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # Loops over vertical axis of the output volume.
            for w in range(n_W):               # Loops over horizontal axis of the output volume.
                for c in range(n_C):           # Loops over the channels of the output volume.
                    
                    # Finds the corners of the current "slice".
                    vert_start = h*stride
                    vert_end = h*stride+f
                    horiz_start = w*stride
                    horiz_end = w*stride+f
                    
                    # Uses the corners to define the slice from a_prev_pad.
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Updates gradients for the window and the filter's parameters.
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Sets the ith training example's dA_prev to the unpadded da_prev_pad.
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad,:]
    
    # Making sure output shape is correct.
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


### Helper Functions for Pool Backward Propagation
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x, for Max Pool Back Propagation.
    
    Arguments:
    x -- Array of shape (f, f).
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """

    mask=np.zeros(x.shape, dtype=bool)
    xmax=np.max(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i,j]==xmax).any():
                mask[i,j]=True
    
    return mask



def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape, for Average Pool Back Propagation.
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz.
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz.
    """

    # Retrieves dimensions from shape.
    (n_H, n_W) = shape
    
    # Computes the value to distribute on the matrix.
    average = dz/(n_H+n_W)
    
    # Creates a matrix where every entry is the "average" value.
    a = np.full((n_W,n_H),average)

    
    return a


### Pool Backward Propagation
def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer.
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A.
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters. 
    mode -- the pooling mode that would like to be used, defined as a string ("max" or "average").
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev.
    """
    
    # Retrieves information from cache.
    (A_prev, hparameters) = cache
    
    # Retrieves hyperparameters from "hparameters".
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieves dimensions from A_prev's shape and dA's shape.
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initializes dA_prev with zeros.
    dA_prev = np.zeros((A_prev.shape))
    
    for i in range(m):                       # Loops over the training examples.
        
        # Selects training example from A_prev.
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # Loops on the vertical axis.
            for w in range(n_W):               # Loops on the horizontal axis.
                for c in range(n_C):           # Loops over the channels (depth).
                    
                    # Finds the corners of the current "slice".
                    vert_start = h*stride
                    vert_end = h*stride+f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    # Computes the backward propagation in both modes.
                    if mode == "max":
                        
                        # Uses the corners and "c" to define the current slice from a_prev.
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        # Creates the mask from a_prev_slice.
                        mask = create_mask_from_window(a_prev_slice)
                        # Sets dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA).
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask*dA[i,h,w,c]
                        
                    elif mode == "average":
                        
                        # Gets the value a from dA.
                        da = dA[i,h,w,c]
                        # Defines the shape of the filter as fxf.
                        shape = (f,f)
                        # Distributes it to get the correct slice of dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da,shape)

    # Making sure your output shape is correct.
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev






def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image.
    n_W0 -- scalar, width of an input image.
    n_C0 -- scalar, number of channels of the input.
    n_y -- scalar, number of classes.
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float".
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float".
    """

    X = tf.placeholder(tf.float32,(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32,(None, n_y))
    
    return X, Y



def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Note that this hard codes the shape values in the function.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2.
    """

    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer())

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters




### FORWARD PROPAGATION
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED.
    
    Note that this hard-codes the stride and kernel (filter) sizes. 
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples).
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters.

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieves the parameters from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'.
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'.
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'.
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU.
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'.
    P2 = P1 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN.
    F = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function.
    # 6 neurons in output layer.
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6).
    Y -- "true" labels vector placeholder, same shape as Z3.
    
    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    return cost



### MODEL FUNCTION
def Model_CNN(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED.
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3).
    Y_train -- test set, of shape (None, n_y = 6).
    X_test -- test set, of shape (None, 64, 64, 3).
    Y_test -- test set, of shape (None, n_y = 6).
    learning_rate -- learning rate of the optimization.
    num_epochs -- number of epochs of the optimization loop.
    minibatch_size -- size of a minibatch.
    print_cost -- True to print the cost every 100 epochs.
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train).
    test_accuracy -- real number, testing accuracy on the test set (X_test).
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # To reset tf variables.
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost.
    
    # Creates Placeholders of the correct shape.
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initializes parameters.
    parameters = initialize_parameters()
    
    # Forward propagation: Builds the forward propagation in the tensorflow graph.
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Adds cost function to tensorflow graph.
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Defines the tensorflow optimizer. Uses an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initializes all the variables globally.
    init = tf.global_variables_initializer()
     
    # Starts the session to compute the tensorflow graph.
    with tf.Session() as sess:
        
        # Runs the initialization.
        sess.run(init)
        
        # Does the training loop.
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set.
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Selects a minibatch.
                (minibatch_X, minibatch_Y) = minibatch
        
                # Runs the graph on a minibatch and the session to execute the optimizer and the cost.
                _ , temp_cost  = sess.run(fetches=[optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Prints the cost every 5 epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # Plots the cost.
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculates the correct predictions.
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculates accuracy on the test set.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("\nTrain Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters
    
    
    


