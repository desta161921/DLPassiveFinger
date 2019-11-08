#!/usr/bin/python3

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

def train_Deep_MLP(fn):
    """
    Function that will train, validate and store statistics of a deep MLP model.
    """
    #------------- PREPARING THE DATA -----------------
    #Feature scaler
    scaler = StandardScaler()

    #Label encoder
    encoder = LabelEncoder()

    #Dataframe for unique labels
    unique_labels = pd.read_csv("input/{}_test_y.csv".format(fn), sep=";")
    unique_labels = [label for label in unique_labels.iloc[:, 0].unique()]

    #Load Training Features
    training_features = pd.read_csv("input/{}_train_X.csv".format(fn), sep=";")
    training_features = training_features.values
    training_features = training_features[:].astype(float)
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)

    #Load Training Labels (Classes)
    training_labels = pd.read_csv("input/{}_train_y.csv".format(fn), sep=";")
    training_labels = training_labels.values
    training_labels = training_labels[:]
    encoder.fit(training_labels.ravel())
    training_labels = encoder.transform(training_labels.ravel())
    training_labels = np_utils.to_categorical(training_labels)

    #Load Testing Features
    testing_features = pd.read_csv("input/{}_test_X.csv".format(fn), sep=";")
    testing_features = testing_features.values
    testing_features = testing_features[:].astype(float)
    scaler.fit(testing_features)
    testing_features = scaler.transform(testing_features)

    #Load Testing Labels (Classes)
    testing_labels = pd.read_csv("input/{}_test_y.csv".format(fn), sep=";")
    testing_labels = testing_labels.values
    testing_labels = testing_labels[:]

    #-------------(END)-------------------

    #---------- DEFINING THE MODEL --------------------

    #Model Variables
    number_of_labels = len(unique_labels)
    nodes = 150
    hidden_layers = 1
    activation_function = "tanh"
    input_dimension = 3 if "NCCM" in fn else 4
    nodes_in_hidden = int(2/3 * (nodes + number_of_labels))
    loss_function = "categorical_crossentropy"
    optimizer_function = "nadam"
    batch_size = 32
    epochs = 150

    #Model definition

    #Type
    model = Sequential()

    #Input layer
    model.add(Dense(nodes,
                    input_dim=input_dimension,
                    activation=activation_function))

    #Hidden layer(s)
    for i in range(hidden_layers):
        model.add(Dense(nodes_in_hidden,
                        activation=activation_function))

    #Output layer
    model.add(Dense(number_of_labels,
                    activation="softmax"))

    #Name the model
    NAME = "Deep_MLP-{}".format(int(time.time()))

    #Callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    model_checkpoint = ModelCheckpoint("models/{}".format(NAME),
                                        monitor="acc",
                                        save_best_only=True,
                                        mode="auto",
                                        period=1)

    #Compile model
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=["acc"])

    #---------------- (END) ----------------

    #---------------- TRAINING AND VALIDATION ------------------
    #Train the model, validation is performed later to be able to create
    #a confusion matrix and a classification report
    model.fit(training_features,
              training_labels,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[tensorboard, model_checkpoint])

    #Predict using the newly trained model
    label_predictions = encoder.inverse_transform(model.predict_classes(testing_features))

    #Making the confusion matrix
    cm = confusion_matrix(testing_labels, label_predictions)

    #Plot confusion matrix
    plot_confusion_matrix(cm, sorted(unique_labels), False)

    #Show the plot and save the figure
    plt.savefig("figures/Deep_MLP_confusion_matrix_{}.svg".format(int(time.time())))
    #plt.show()


    #Write a .txt report file
    with open("reports/Deep_MLP_{}_report.txt".format(fn), "w") as f:
        f.write("REPORT FOR \"{}\"\n\n".format(fn))

        f.write("\n\n\nClassification Report:\n")
        for line in classification_report(testing_labels, label_predictions):
            f.write(line)

        f.write("\nConfusion Matrix:\n\n")
        f.write(np.array2string(cm, separator=', '))

        f.write("\n\nScore for final model:\n")
        f.write("Accuracy: {}".format(accuracy_score(testing_labels,
                                                     label_predictions)))

        f.close()

    #-------------- (END) -------------------

if __name__ == '__main__':
    files = ["No_R", "No_R_NCCM", "Full_R", "Full_R_NCCM"]

    train_Deep_MLP(files[2])
