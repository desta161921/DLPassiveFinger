import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler

def test_RF(fn):
    """
    Function which will tune and test a Random Forest model. It will plot
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

    #Fitting Random Forest Classifier to the Training set
    rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=7)
    start = time.time()
    rf.fit(X_train, encoded_y_train)
    end = time.time()

    #Predicted values
    y_pred = encoder.inverse_transform(rf.predict(X_test))

    #Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n")
    print(classification_report(y_test, y_pred))
    print("Scores for final, best model:\n")
    print("Acc: {}".format(accuracy_score(y_test, y_pred)))

    #Find labels
    labels = [label for label in y_test_df.iloc[:, 0].unique()]

    #Plot confusion matrix
    plot_confusion_matrix(cm, sorted(labels), False)

    #Show the plot
    plt.savefig("figures/RF_confusion_matrix_{}.svg".format(int(time.time())))
    #plt.show()


    #Write a .txt report file
    with open("reports/RF_{}_report.txt".format(fn), "w") as f:
        f.write("REPORT FOR \"{}\"\n\n".format(fn))

        f.write("\n\n\nClassification Report:\n")
        for line in classification_report(y_test, y_pred):
            f.write(line)

        f.write("\nConfusion Matrix:\n\n")
        f.write(np.array2string(cm, separator=', '))


        f.write("\n\nTime used to train the model: {} seconds".format(end-start))

        f.write("\n\nScores for final, best model:\n")
        f.write("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

        f.close()


if __name__ == '__main__':
    files = ["No_R", "No_R_NCCM", "Full_R", "Full_R_NCCM"]

    test_RF(files[2])
