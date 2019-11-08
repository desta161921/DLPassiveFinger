import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler

def test_KNN(fn):
    """
    Function which will tune and test a K-Nearest Neighbors model. It will plot
    a confusion matrix and write a performance report to file.

    Arguments:
        - fn        :       Name of the input file.
    """
    #Timer variables
    start = 0
    end = 0

    #Load datasets
    X_train_df = pd.read_csv("input/{}_train_X.csv".format(fn), sep=";")
    y_train_df = pd.read_csv("input/{}_train_y.csv".format(fn), sep=";")
    X_test_df = pd.read_csv("input/{}_test_X.csv".format(fn), sep=";")
    y_test_df = pd.read_csv("input/{}_test_y.csv".format(fn), sep=";")

    X_val_tr = X_train_df.values
    y_val_tr = y_train_df.values
    X_val_test = X_test_df.values
    y_val_test = y_test_df.values

    #Convert to numpy arrays
    X_train = X_val_tr[:].astype(float)
    y_train = y_val_tr[:]
    X_test = X_val_test[:].astype(float)
    y_test = y_val_test[:]

    #Scale X values (train)
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    #Scale X values (test)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    #Transform non-numerical values into numericals
    encoder = LabelEncoder()
    encoder.fit(y_train.ravel())
    encoded_y_train = encoder.transform(y_train.ravel())
    encoder.fit(y_test.ravel())
    encoded_y_test = encoder.transform(y_test.ravel())

    #Number of neighbors (K) to test
    nr_of_neighbors = [x for x in range(5,100,5)]

    #Variables to store the best values
    best_model = KNeighborsClassifier()
    best_acc = 0.0
    time_taken = 0

    #Test different values for K
    for K in nr_of_neighbors:
        knn = KNeighborsClassifier(n_neighbors=K)

        #Train the model
        start = time.time()
        knn.fit(X_train, encoded_y_train)
        end = time.time()

        #Predicted values
        y_pred = knn.predict(X_test)

        print("\nK: {}".format(knn.get_params()['n_neighbors']))
        print("Acc: {}".format(accuracy_score(encoded_y_test, y_pred)))

        #Measure accuracy and save model if it is the best one
        if accuracy_score(encoded_y_test, y_pred) > best_acc:
            time_taken = end - start
            best_model = knn
            best_acc = accuracy_score(encoded_y_test, y_pred)


    #Predict using the best model
    y_pred = encoder.inverse_transform(best_model.predict(X_test))
    K = best_model.get_params()['n_neighbors']

    #Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n")
    print(classification_report(y_test, y_pred))
    print("Scores for final, best model:\n")
    print("\nK: {}".format(K))
    print("Acc: {}".format(accuracy_score(y_test, y_pred)))

    #Find labels
    labels = [label for label in y_test_df.iloc[:, 0].unique()]

    #Plot confusion matrix
    plot_confusion_matrix(cm, sorted(labels), False)

    #Show the plot
    plt.savefig("figures/KNN_confusion_matrix_{}.svg".format(int(time.time())))
    #plt.show()

    #Write a .txt report file
    with open("reports/KNN_{}_report.txt".format(fn), "w") as f:
        f.write("REPORT FOR \"{}\"\n\n".format(fn))
        f.write("Best value for K: {}".format(K))

        f.write("\n\n\nClassification Report:\n")
        for line in classification_report(y_test, y_pred):
            f.write(line)

        f.write("\nConfusion Matrix:\n\n")
        f.write(np.array2string(cm, separator=', '))

        f.write("\n\nTime used to train the model: {} seconds".format(time_taken))

        f.write("\n\nScores for final, best model:\n")
        f.write("Accuracy: {}".format(best_acc))

        f.close()



if __name__ == '__main__':
    files = ["No_R", "No_R_NCCM", "Full_R", "Full_R_NCCM"]

    test_KNN(files[2])
