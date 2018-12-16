
"""
MNIST Example
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import csv as csv
import pandas as pd
import sys


"""
Local Variables
"""
#All Data
UData = 0
#Training and Testing Data
trainX = trainY = testX = testY = 0
#Number of categories for each non-numeric attribute
num_distances = num_classes = num_gaits = num_track_conditions = num_medicines = 0
#DNN
classifier = 0
#ID of which data to split the training and testing data
id_split = "2017032550"
#Weights to weight winning horses higher
train_weights = test_weights = 0
#Input for the neural network with all columns being numerical
feature_columns = weight_column = 0


"""
Main Function
"""
def main():

  #clarifies which global variables will be used
  global UData
  global feature_columns
  global weight_column
  global train_weights
  global test_weights

  #reads in data
  UData = read_in_data("Data/CleanData.csv")

  #itimizes all the non-numerical categories into text files
  list_types_in_attributes(UData)

  #converts the place column to a win column
  UData = convert_place_to_win(UData)

  #splits train and test data
  split_train_vs_test(UData)

  #weights the wins in order to incentives the neural network to predict them
  (train_weights, test_weights) = weight_win_data(UData)

  #builds inputs to the neural network by adding weights and making all columns numerical
  feature_columns, weight_column = make_feature_columns()

  #builds and runs a neural network
  print(test(1, .01, 50, 10, "Adam"))



"""
This tests a neural network taking the following arguments
epochs: number of training iterations
lr: learning rate
num_hidden_layers: number of hidden layers in nueral network
num_neurons_per_layer: number of neurons per layer
Optimizer: Either "Adam" or "GradientDescent"
"""
def test(epochs, lr, num_hidden_layers, num_neurons_per_layer, optimizer):
    global UData
    global feature_columns
    global weight_column
    global train_weights
    global test_weights

    #makes neural network
    make_neural_network(feature_columns, weight_column, lr, num_hidden_layers, num_neurons_per_layer, optimizer)
    
    #trains neural network recording training loss
    training_loss = train_neural_network(train_weights, epochs)

    #tests neural network recording testing loss and predictions
    testing_loss, test_input_fn = test_neural_network(test_weights)

    #evaluates neural network performance if it bet for a year
    final_bank = evaluate_bets(UData, test_input_fn)

    return (training_loss, testing_loss, final_bank)


"""
Reads in file with Data
"""
def read_in_data(filename):
    with open(filename, 'rb') as f:
        mycsv = csv.reader(f)
        mycsv = list(mycsv)
        
    UData = np.array(mycsv)
    return UData

"""
Lists the categories in each non-numberic attribute into a text file
This allows tensorflow to convert these categories into different types of feature columns
"""
def list_types_in_attributes(UData):
    global num_distances 
    global num_classes 
    global num_gaits 
    global num_track_conditions
    global num_medicines
    categ = [1,2,4,5,7]
    total_vocab = []
    num_vocab = []
    for y in categ:
        vocab = []
        for x in range(1, len(UData)):
            if UData[x][y] not in vocab:
                vocab.append(UData[x][y])
        total_vocab.append(vocab)
        num_vocab.append(len(vocab))


    var_num = 0
    text_file = open("Data/Distances.txt", "w")
    num_distances = num_vocab[var_num]
    for y in range(num_distances):
        text_file.write("%s\n" % total_vocab[var_num][y])
        
    text_file.close()

    var_num = 1
    text_file = open("Data/Classes.txt", "w")
    num_classes = num_vocab[var_num]
    for y in range(num_classes):
        text_file.write("%s\n" % total_vocab[var_num][y])
    text_file.close()

    var_num = 2
    text_file = open("Data/Gaits.txt", "w")
    num_gaits = num_vocab[var_num]
    for y in range(num_gaits):
        text_file.write("%s\n" % total_vocab[var_num][y])
    text_file.close()

    var_num = 3
    text_file = open("Data/Track Conditions.txt", "w")
    num_track_conditions = num_vocab[var_num]
    for y in range(num_track_conditions):
        text_file.write("%s\n" % total_vocab[var_num][y])
    text_file.close()

    var_num = 4
    text_file = open("Data/Medicines.txt", "w")
    num_medicines = num_vocab[var_num]
    for y in range(num_medicines):
        text_file.write("%s\n" % total_vocab[var_num][y])
    text_file.close()




"""
This splits training data from testing data
"""
def split_train_vs_test(UData):
    global trainX 
    global trainY
    global testX
    global testY
    trainX = np.array([x for x in UData[1:,:-7] if train(x)])
    testX = np.array([x for x in UData[1:,:-7] if not train(x)]) 
    trainY = np.array([win(y) for y in UData[1:,(0,-1)] if train(y)])
    testY = np.array([win(y) for y in UData[1:,(0,-1)] if not train(y)])

"""
Weights the winning horse data higher. This prevents the neural network from
just predicting all horse lose. 
"""
def weight_win_data(UData):
    global trainX 
    global trainY
    global testX
    global testY
    zero_weight = .18
    one_weight = .82
    train_weights = np.array(trainY[:,1]).astype(float)
    for x in range(len(trainY)):
        if train_weights[x] == 0:
            train_weights[x] = zero_weight
        elif train_weights[x] == 1:
            train_weights[x] = one_weight
        else:
            print("\n\nERROR\n\n")
            
    test_weights = np.array(testY[:,1]).astype(float)
    for x in range(len(testY)):
        if test_weights[x] == 0:
            test_weights[x] = zero_weight
        elif test_weights[x] == 1:
            test_weights[x] = one_weight
        else:
            print("\n\nERROR\n\n")

    return (train_weights, test_weights)
      
"""
Converts all feature columns into inputs feasable for the neural network
"""
def make_feature_columns():
    global num_distances 
    global num_classes 
    global num_gaits 
    global num_track_conditions
    global num_medicines
    weight_column = tf.feature_column.numeric_column(key = 'weight')

    distance_col = tf.feature_column.categorical_column_with_vocabulary_file(key="Distance", vocabulary_file="Data/Distances.txt",
                                                                                    vocabulary_size=num_distances)
    class_col = tf.feature_column.categorical_column_with_vocabulary_file(key="Class", vocabulary_file="Data/Classes.txt",
                                                                                    vocabulary_size=num_classes)
    gait_col  = tf.feature_column.categorical_column_with_vocabulary_file(key="Gait", vocabulary_file="Data/Gaits.txt",
                                                                                     vocabulary_size=num_gaits)
    track_condition_col = tf.feature_column.categorical_column_with_vocabulary_file(key="TC", vocabulary_file="Data/Track Conditions.txt",
                                                                                    vocabulary_size=num_track_conditions)
    post_position_col = tf.feature_column.categorical_column_with_identity(key="PP", num_buckets=10)
    new_horse_col = tf.feature_column.categorical_column_with_identity(key="NewH", num_buckets=2)
    medicine_col  = tf.feature_column.categorical_column_with_vocabulary_file(key="Medicine", vocabulary_file="Data/Medicines.txt",
                                                                                    vocabulary_size=num_medicines)
    feature_columns = [tf.feature_column.indicator_column(distance_col),
                       tf.feature_column.indicator_column(class_col),
                       tf.feature_column.numeric_column(key="NH"),
                       tf.feature_column.indicator_column(gait_col),
                       tf.feature_column.indicator_column(track_condition_col),
                       tf.feature_column.indicator_column(post_position_col),
                       tf.feature_column.indicator_column(medicine_col),
                       tf.feature_column.numeric_column(key="1F"),
                       tf.feature_column.numeric_column(key="2F"),
                       tf.feature_column.numeric_column(key="3F"),
                       tf.feature_column.indicator_column(new_horse_col),
                       tf.feature_column.numeric_column(key="HWP"),
                       tf.feature_column.numeric_column(key="JWP"),
                       tf.feature_column.numeric_column(key="TWP")
                      ]

    return (feature_columns, weight_column)

"""
We used a classfier for the neural network from tensorflow estimators. We first built a more in depth version, but realized that it just optimized by saying all
the horses lose every race. In order to combat this, we had to add a weight column in order to incentives it to predict winning horses. This could only be done 
 by using a DNN from a tensorflow estimator. 
"""
def make_neural_network(feature_columns, weight_column, lr, num_hidden_layers, num_neurons_per_layer, optimizer):


    # Build 2 layer DNN classifier
    global classifier

    if optimizer == "Adam":
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=np.full((num_hidden_layers),num_neurons_per_layer),
            optimizer=tf.train.AdamOptimizer(lr),
            n_classes=2,
            weight_column = weight_column,
        )
    elif optimizer == "GradientDescent":
        classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=np.full((num_hidden_layers),num_neurons_per_layer),
            optimizer=tf.train.GradientDescentOptimizer(lr),
            n_classes=2,
            weight_column = weight_column
        )
    else:
      nERROR("Optimizer not found")

"""
Trains the neural network 
"""
def train_neural_network(train_weights, epochs):
    global classifier

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"weight": train_weights,
           "Distance": trainX[:,1], 
           "Class": trainX[:,2], 
           "NH": trainX[:,3].astype(int), 
           "Gait": trainX[:,4], 
           "TC": trainX[:,5],
           "PP": trainX[:,6].astype(int), 
           "Medicine": trainX[:,7], 
           "1F": trainX[:,8].astype(int), 
           "2F": trainX[:,9].astype(int), 
           "3F": trainX[:,10].astype(int), 
           "NewH": trainX[:,11].astype(int), 
           "HWP": trainX[:,12].astype(float), 
           "JWP": trainX[:,13].astype(float), 
           "TWP": trainX[:,14].astype(float), 
           },
        y=trainY[:,-1].astype(int),
        num_epochs=epochs,
        shuffle=True,
    )

    classifier.train(input_fn = train_input_fn)


    return classifier.evaluate(input_fn=train_input_fn)["average_loss"]


"""
Tests the neural network
"""
def test_neural_network(test_weights):
    global classifier
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
         x={"weight": test_weights,
            "Distance": testX[:,1], 
           "Class": testX[:,2], 
           "NH": testX[:,3].astype(int), 
           "Gait": testX[:,4], 
           "TC": testX[:,5],
           "PP": testX[:,6].astype(int), 
           "Medicine": testX[:,7], 
           "1F": testX[:,8].astype(int), 
           "2F": testX[:,9].astype(int), 
           "3F": testX[:,10].astype(int), 
           "NewH": testX[:,11].astype(int), 
           "HWP": testX[:,12].astype(float), 
           "JWP": testX[:,13].astype(float), 
           "TWP": testX[:,14].astype(float), 
           },
        y=testY[:,-1].astype(int),
        num_epochs=1,
        shuffle=False
    )
    average_loss = classifier.evaluate(input_fn=test_input_fn)["average_loss"]
    return average_loss, test_input_fn


"""
Evaluates the model for all of the testing data in terms of betting.
Bets $10 per bet
"""
def evaluate_bets(UData, test_input_fn):
    prob_bet = .8
    ds_predict_tf = classifier.predict(input_fn = test_input_fn)
    z = 0
    bank = 0
    bank_history = []
    bank_history.append(bank)
    for i in ds_predict_tf:
        if i['probabilities'][1] > prob_bet:
            bank -= 2
            if int(testY[z][1]) == 1:
                temp = [y for y in UData if testY[z][0] == y[0]]
                bank += float(temp[0][-7])
            bank_history.append(bank)
        z += 1


    import matplotlib.pyplot as plt
    plt.plot(bank_history)
    plt.show()
    return bank


"""
Returns 1 if horse won the race and 0 otherwise
"""
def win(x):
    if int(x[-1]) == 1:
        x[-1] = 1
    else:
        x[-1] = 0
    return x

"""
Returns True if it was passed an ID that should be part of the training data, false otherwise
"""
def train(x):
    if int(x[0]) < int(id_split):
        return True
    return False

"""
Converts the place column to a win column
"""
def convert_place_to_win(UData):
    return np.vstack((UData[0], [win(x) for x in UData[1:]]))


main()





