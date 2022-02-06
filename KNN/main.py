#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from knn import KNN
import utils


@handle("1")
def classifierPlot():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    # k = 1
    # k = 3
    k = 10

    model = KNN(k)
    """YOUR CODE HERE FOR Q1"""
    model.fit(X, y)
    y_pred = model.predict(X_test)

    # Error
    error = np.mean(y_pred != y_test)
    print(f"Knn-Classification with k = {k} error: {error:.3f}")

    plot_classifier(model, X, y)

    plot_name = "kNN_Classify_Plot" + "_k=" + str(k) + ".pdf"

    fname = Path("plots", plot_name)
    print(fname)
    plt.savefig(fname)
    print(f"\nFigure saved as {fname}")

@handle("2")
def crossValidation():

    ## j-fold Cross Validation
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]

    n,d = X.shape
    j = 10
    jth_of_data = int(n/10)

    ks = list(range(1, 30, 4))

    cv_accs = np.zeros(len(ks))

    for k in ks:
        model = KNN(k)

        mask = np.ones(n, dtype = bool)
        cv_quantile = 0    ## will increment it as we go

        each_fold_acc = np.zeros(10)

        while (cv_quantile < 10):
            mask[cv_quantile * jth_of_data : (cv_quantile+1) * jth_of_data - 1] = 0
            x_train = X[mask,]
            y_train = y[mask]
            x_validation = X[~mask,]
            y_validation = y[~mask]

            model.fit(x_train, y_train)
            y_validation_pred = model.predict(x_validation)
            accuracy = np.mean(y_validation_pred == y_validation)

            each_fold_acc[cv_quantile] = accuracy
            mask[cv_quantile * jth_of_data : (cv_quantile+1) * jth_of_data - 1] = 1

            cv_quantile += 1

        avg_accuracy = each_fold_acc.mean()
        cv_accs[ks.index(k)] = avg_accuracy

    plt.plot(ks, cv_accs, label="cv_accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Cross Validation Accuracy")

    fname = Path("plots", "CV_KvsAccuracyPlot.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

@handle("3")
def test_CV_accuracy():

    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    n, d = X.shape
    jth_of_data = int(n / 10)

    ks = list(range(1, 30, 4))
    mask = np.ones(n, dtype=bool)

    cv_accs = np.zeros(len(ks))
    test_accs = np.zeros(len(ks))

    for k in ks:
        model = KNN(k)

        ## CV
        mask = np.ones(n, dtype=bool)
        cv_quantile = 0  ## will increment it as we go

        each_fold_acc = np.zeros(10)

        while (cv_quantile < 10):
            mask[cv_quantile * jth_of_data: (cv_quantile + 1) * jth_of_data - 1] = 0
            x_train = X[mask,]
            y_train = y[mask]
            x_validation = X[~mask,]
            y_validation = y[~mask]

            model.fit(x_train, y_train)
            y_validation_pred = model.predict(x_validation)
            accuracy = np.mean(y_validation_pred == y_validation)

            each_fold_acc[cv_quantile] = accuracy
            mask[cv_quantile * jth_of_data: (cv_quantile + 1) * jth_of_data - 1] = 1

            cv_quantile += 1

        avg_accuracy = each_fold_acc.mean()
        cv_accs[ks.index(k)] = avg_accuracy
        # print(f"Added CV accuracy for k = {k}: Accuracy = {avg_accuracy}")

        ## TESTING ACC
        model.fit(X, y)
        y_pred = model.predict(X_test)

        accuracy = np.mean(y_pred == y_test)
        test_accs[ks.index(k)] = accuracy
        # print(f"Added testing accuracy for k = {k}: Accuracy = {accuracy}")

    plt.plot(ks, test_accs, label="test_accuracy")
    plt.plot(ks, cv_accs, label="cv_accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Classification Accuracy")
    plt.legend()

    fname = Path("plots", "CV_TestAccuracy_Vs_K_Plot.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)

@handle("4")
def training_error():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]

    ks = list(range(1, 30, 4))

    train_errors = np.zeros(len(ks))

    for k in ks:
        model = KNN(k)
        model.fit(X, y)
        y_pred = model.predict(X)
        error = np.mean(y != y_pred)
        train_errors[ks.index(k)] = error
        # print(f"Added training error for k = {k}: Error = {error}")

    plt.plot(ks, train_errors, label="trainerror")
    plt.xlabel("Value of K")
    plt.ylabel("Training error")

    fname = Path("plots", "TrainingErrorVsKPlot.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


if __name__ == "__main__":
    main()
