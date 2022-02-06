import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder #to convert strings into ints
from keras.utils import to_categorical

import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #solution from the web to get rid of specific error

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

#MANIMUPLATE DATASET

    #  Load dataset
    features = np.load('feature_array.npy') #shape = 1514x40 -> 1514 files and 40 mfccs per file
    classes = np.load('class_array.npy') #shape = 1514x1

    #encoding the string classes into numeric labels
    le = LabelEncoder()
    encoded_labels = to_categorical(le.fit_transform(classes))

    #tansfer matrix from (x,10) into a (x,) so the 0 1 0 0 enoding becomes one column
    x = np.array([0,1,2,3,4,5,6,7,8,9])
    labels = np.matmul(encoded_labels,x)

    # split the dataset : x=features, y = labels
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=127)

#BUILD AND RUN THE MODEL
    #Building the model - sequential (feed-forward)
    model = keras.Sequential([
        keras.layers.Dense(40, input_shape=(40,)),  # input layer (1)
        keras.layers.Dense(256, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(10, activation='softmax')  # output layer (3)
    ])

    #Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #Training the model
    model.fit(x_train, y_train, epochs=10)  # we pass the data, labels and epochs and watch the magic!

    #Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', test_acc)










