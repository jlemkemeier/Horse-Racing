
"""
MNIST Example
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import csv as csv
import pandas as pd
import sys




#read csv
with open("Data/CleanData.csv", 'rb') as f:
    mycsv = csv.reader(f)
    mycsv = list(mycsv)
    
UData = np.array(mycsv)

print(len(UData))
print(UData[0])




print(UData[3:,-1])




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
print(total_vocab)
print(num_vocab)

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




def win(x):
    if int(x[-1]) == 1:
        x[-1] = 1
    else:
        x[-1] = 0
    return x




id_split = "2017032550"
id_split[:4]




def train(x):
    if int(x[0]) < int(id_split):
        return True
    return False




#Convert Place to win 
UData = np.vstack((UData[0], [win(x) for x in UData[1:]]))


# In[9]:



trainX = np.array([x for x in UData[1:,:-7] if train(x)])
testX = np.array([x for x in UData[1:,:-7] if not train(x)]) 
trainY = np.array([win(y) for y in y if train(y)])
testY = np.array([win(y) for y in UData[1:,(0,-1)] if not train(y)])
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))




#X = np.delete(UData, -1, axis=1)[1:-1]
#Y = [win(x) for x in UData[1:-1,-1]]
#print('tX', X[0])
#print('tY', Y[0])
#print(len(X))
#print(len(Y))




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
    




#import tensorflow as tf
#import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data')

#def input(dataset):
#    return dataset

# Specify feature
#feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
print("1Here")
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
print("4Here")


# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=np.full((100),5),
    optimizer=tf.train.AdamOptimizer(1e-5),
    n_classes=2,
    weight_column = weight_column
    #warm_start_from = "/var/folders/tx/t5mjmx9d5vvg8r4r62w32mg80000gn/T/tmpINcd1R/"
)
print("5Here")


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
    num_epochs=50000,
    shuffle=True,
)
print("6Here")


classifier.train(input_fn = train_input_fn, steps=50000)
print("7Here")


def test_set():
    return np.array(nX[:,1], float32)
print("8Here")

# Define the test inputs
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
    
    


# Evaluate accuracy
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))
print("10Here")






prob_bet = .4

ds_predict_tf = classifier.predict(input_fn = test_input_fn)
print('Predictions: {}'.format(str(ds_predict_tf)))
z = 0
bank = 0
bank_history = []
bank_history.append(bank)
for i in ds_predict_tf:
    #print("predict")
    if i['probabilities'][1] > prob_bet:
        bank -= 2
        if int(testY[z][1]) == 1:
            temp = [y for y in UData if testY[z][0] == y[0]]
            
            bank += float(temp[0][-7])
            print("\nWINNING BET")
            print("Neural Net Probability", i['probabilities'][1])
            print(temp)
            print("odds", temp[0][-7])
        else:
            print("\nLOSING BET")
            print("Neural Net Probability", i['probabilities'][1])
        bank_history.append(bank)

            
            
        
        print("New amount", bank)
    z += 1
print("final",bank)




import matplotlib.pyplot as plt
plt.plot(bank_history) # plotting by columns

plt.show()




