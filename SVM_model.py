import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler

def test_SVM(fn):
    """
    Function which will tune and test a Support Vector Machine model. It will
    plot a confusion matrix and write a performance report to file.

    Arguments:
        - fn        :       Name of the input file.
    """
    ##### INTRO #####
    #Timer variables
    star = 0
    end  = 0
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

    ##### TUNING #####

    #Variables
    Cs = [1, 10, 100, 1000, 1500]
    gammas = [0.001, 0.01, 0.1, 1, 10, 15]
    degrees = [0, 1, 2, 3, 4, 5, 6]
    # Create the parameter grid based on the results of random search
    params_grid = [#{'C': Cs, 'kernel':['linear']},
                   {'C': Cs, 'gamma': gammas, 'kernel': ['rbf']}]
                   #{'C': Cs, 'gamma': gammas, 'kernel': ['poly']}]

    # Performing CV to tune parameters for best SVM fit
    start = time.time()
    svm_model = GridSearchCV(SVC(), params_grid, cv=5, verbose=2, n_jobs=4)
    svm_model.fit(X_train, encoded_y_train)
    end = time.time()

    # View the accuracy score
    print('Best score for training data:', svm_model.best_score_,"\n")

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n")
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_

    print("Grid scores on development set:\n")
    means = svm_model.cv_results_['mean_test_score']
    stds = svm_model.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svm_model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    ##### VALIDATION #####

    predictions = final_model.predict(X_test)
    predicted_labels = list(encoder.inverse_transform(predictions))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, predicted_labels)
    print(cm)
    print("\n")
    print(classification_report(y_test,predicted_labels))

    #Find labels
    labels = [label for label in y_test_df.iloc[:, 0].unique()]

    #Plot confusion matrix
    plot_confusion_matrix(cm, sorted(labels), False)

    #Show the plot
    plt.savefig("figures/SVM_confusion_matrix_{}.svg".format(int(time.time())))
    #plt.show()

    print("Training set score for SVM: %f" % final_model.score(X_train, encoded_y_train))
    print("Testing  set score for SVM: %f" % final_model.score(X_test, encoded_y_test))

    svm_model.score

    ##### WRITING REPORT #####

    with open("reports/SVM_{}_report.txt".format(fn), "w") as f:
        f.write("REPORT FOR \"{}\"\n\n".format(fn))

        f.write("Best parameters for training data and score:\n")
        f.write("Score: {}\n".format(svm_model.best_score_))
        f.write("C: {}\n".format(svm_model.best_estimator_.C))
        f.write("Gamma: {}\n".format(svm_model.best_estimator_.gamma))
        f.write("Kernel: {}\n".format(svm_model.best_estimator_.kernel))

        f.write("\n\n\nClassification Report:\n")
        for line in classification_report(y_test,predicted_labels):
            f.write(line)

        f.write("\nConfusion Matrix:\n\n")
        f.write(np.array2string(cm, separator=', '))


        f.write("\n\nTime used to train the model: {} seconds".format(end-start))

        f.write("\n\nScores for final, best model:\n")
        f.write("Score for training set: {}\n".format(final_model.score(X_train, encoded_y_train)))
        f.write("Score for testing set: {}".format(final_model.score(X_test, encoded_y_test)))

        f.close()

if __name__ == '__main__':
    files = ["No_R", "No_R_NCCM", "Full_R", "Full_R_NCCM"]

    test_SVM(files[2])
