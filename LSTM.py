#!/usr/bin/python3

from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import time


def train_LSTM_model(fn):
    """
    Function which will train a LSTM-model and plot its statistics.
    """

    #Feature scaler
    scaler = MinMaxScaler()

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
    training_features = np.reshape(training_features,
                                   (training_features.shape[0],
                                   1,
                                   training_features.shape[1]))

    #Load Testing Features
    testing_features = pd.read_csv("input/{}_test_X.csv".format(fn), sep=";")
    testing_features = testing_features.values
    testing_features = testing_features[:].astype(float)
    scaler.fit(testing_features)
    testing_features = scaler.transform(testing_features)
    testing_features = np.reshape(testing_features,
                                  (testing_features.shape[0],
                                  1,
                                  testing_features.shape[1]))

    #Load Training Labels (Classes)
    training_labels = pd.read_csv("input/{}_train_y.csv".format(fn), sep=";")
    training_labels = training_labels.values
    training_labels = training_labels[:]
    encoder.fit(training_labels.ravel())
    training_labels = encoder.transform(training_labels.ravel())
    training_labels = np_utils.to_categorical(training_labels)

    #Load Testing Labels (Classes)
    testing_labels = pd.read_csv("input/{}_test_y.csv".format(fn), sep=";")
    testing_labels = testing_labels.values
    testing_labels = testing_labels[:]
    # encoder.fit(testing_labels.ravel())
    # testing_labels = encoder.transform(testing_labels.ravel())
    # testing_labels = np_utils.to_categorical(testing_labels)

    #Model Variables
    number_of_labels = len(unique_labels)
    nodes = 150
    hidden_layers = 1
    activation_function = "relu"
    recurrent_activation = "hard_sigmoid"
    dropout = 0.01
    input_shape = (1, 3) if "NCCM" in fn else (1, 4)
    nodes_in_hidden = int(2/3 * (nodes + number_of_labels))
    loss_function = "mse" 
    optimizer_function = "adam"
    batch_size = 32
    epochs = 150

    #Model definition

    #Type
    model = Sequential()

    #Input Layer
    model.add(LSTM(nodes,
                input_shape=input_shape,
                dropout=0.009,
                recurrent_dropout=0.01,
                activation=activation_function,
                recurrent_activation=recurrent_activation,
                return_sequences=True if hidden_layers > 0 else False))
    model.add(BatchNormalization())

    #Hidden Layer(s)
    for i in range(hidden_layers):
        if i == hidden_layers - 1:
            model.add(LSTM(nodes_in_hidden,
                           dropout=dropout,
                           recurrent_dropout=dropout,
                           recurrent_activation=recurrent_activation,
                           activation=activation_function))
            model.add(BatchNormalization())
        else:
            model.add(LSTM(nodes_in_hidden,
                           dropout=dropout,
                           recurrent_dropout=dropout,
                           activation=activation_function,
                           recurrent_activation=recurrent_activation,
                           return_sequences=True))
        model.add(BatchNormalization())

    #Output Layer
    model.add(Dense(number_of_labels, activation="softmax"))

    #Name the model
    NAME = "LSTM-{}".format(int(time.time()))

    #Callbacks
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    model_checkpoint = ModelCheckpoint("models/{}".format(NAME),
                                        monitor="acc",
                                        save_best_only=True,
                                        mode="auto",
                                        period=1)

    #Compile the model
    model.compile(loss=loss_function, optimizer=optimizer_function, metrics=["acc"])

    #Train the model, validation is performed later to be able to create
    #a confusion matrix and a classification report
    history = model.fit(training_features,
              		training_labels,
              		batch_size=batch_size,
              		epochs=epochs,
              		callbacks=[tensorboard, model_checkpoint])

    #Predict using the newly trained model
    label_predictions = encoder.inverse_transform(model.predict_classes(testing_features))

    #Creating the confusion matrix
    cm = confusion_matrix(testing_labels, label_predictions)

    #Plot confusion matrix
    plot_confusion_matrix(cm, sorted(unique_labels), False)

    #Show the plot and save the figure
    plt.savefig("figures/LSTM_confusion_matrix_{}.svg".format(int(time.time())))
    #plt.show()


    #Write a .txt report file
    with open("reports/LSTM_{}_report_{}.txt".format(fn, int(time.time())), "w") as f:
        f.write("REPORT FOR \"{}\"\n\n".format(fn))
        f.write("MODEL VARIABLES:\n\n")
        f.write("Number of nodes:" + "."*34 + "{}\n".format(nodes))
        f.write("Hidden layers:" + "."*36 + "{}\n".format(hidden_layers))
        f.write("Activation function:" + "."*30 + "{}\n".format(activation_function))
        f.write("Recurrent activation function:" + "."*20 + "{}".format(recurrent_activation) + "\n")
        f.write("Dropout:" + "."*42 + "{}".format(dropout) + "\n")
        f.write("Loss function:" + "."*36 + "{}".format(loss_function) + "\n")
        f.write("Optimizer:" + "."*40 + "{}".format(optimizer_function) + "\n")
        f.write("Batch size:" + "."*39 + "{}".format(batch_size) + "\n")

        f.write("\n\n\nClassification Report:\n")
        for line in classification_report(testing_labels, label_predictions):
            f.write(line)

        f.write("\nConfusion Matrix:\n\n")
        f.write(np.array2string(cm, separator=', '))

        f.write("\n\nScore for final model:\n")
        f.write("Training Accuracy: \t{}\n".format(history.history['acc'][-1]))
        f.write("Validation Accuracy: {}".format(accuracy_score(testing_labels,
                                                     label_predictions)))

        f.close()

if __name__ == '__main__':
    files = ["No_R", "No_R_NCCM", "Full_R", "Full_R_NCCM"]

    train_LSTM_model(files[2])
