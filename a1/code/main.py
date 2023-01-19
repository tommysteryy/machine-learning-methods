#!/usr/bin/env python

# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from kmeans import KMeans
from kmedians import KMedians
from utils import (
    test_and_plot,
    plot2Dclassifier,
    plot2Dclusters,
    load_dataset,
    main,
    handle,
    run,
)
from least_squares import LeastSquares, LeastSquaresBias, LeastSquaresRBFL2
from logistic_regression import OneVsRestClassifier, LogisticRegression
from softmax_classifier import SoftmaxClassifier


@handle("5")
def q5():
    (X,) = load_dataset("clusterData", "X")
    model = KMeans(X, 4, plot=True)


def best_fit(X, k, reps=50, cls=KMeans):
    # Fit a cls() reps number of times, and return the best one according to model.loss()
    # Use  cls(X, k, plot=False, log=False)  to fit a model,
    # so it'll work for both KMeans and KMedians.
    # (Passing plot=False makes it run a *lot* faster, and log=False avoids a ton of clutter.)

    raise NotImplementedError()


@handle("5a")
def q_5a():
    (X,) = load_dataset("clusterData", "X")
    model = KMeans(X, 4, plot=False)

@handle("5c")
def q_5c():
    (X,) = load_dataset("clusterData", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(X, best_model.get_assignments(X), best_model.w, "kmeans-best.png")


@handle("5f")
def q_5f():
    (X,) = load_dataset("clusterData2", "X")
    best_model = best_fit(X, k=4)
    plot2Dclusters(X, best_model.get_assignments(X), best_model.w, "kmeans-outliers.png")


@handle("5g")
def q_5g():
    (X,) = load_dataset("clusterData2", "X")
    best_model = best_fit(X, k=4, cls=KMedians)
    plot2Dclusters(X, best_model.get_assignments(X), best_model.w, "kmedians-outliers.png")


@handle("6")
def q_6():
    X, y = load_dataset("basisData", "X", "y")
    model = LeastSquares(X, y)
    test_and_plot(model, X, y, filename="leastsquares.png")


@handle("6a")
def q_6a():
    X, y = load_dataset("basisData", "X", "y")
    model = LeastSquaresBias(X, y)
    test_and_plot(model, X, y, filename="leastsquares-bias.png")


@handle("6b")
def q_6b():
    X, y = load_dataset("basisData", "X", "y")
    model = LeastSquaresRBFL2(X, y)
    test_and_plot(model, X, y, filename="leastsquares-rbfl2.png")


@handle("6c")
def q_6c():
    X, y = load_dataset("basisData", "X", "y")

    raise NotImplementedError()



@handle("7")
def q_7():
    X, y, X_test, y_test = load_dataset("multiData", "X", "y", "Xtest", "ytest")

    ovr_lr = OneVsRestClassifier(LogisticRegression, X, y)
    ovr_lr_y_hat = ovr_lr.predict(X_test)
    print(f"Logistic regression test error: {np.mean(ovr_lr_y_hat != y_test):.1%}")
    plot2Dclassifier(
        ovr_lr, X, y, X_test=X_test, y_test=y_test, k=5, filename="lr-preds.png"
    )

    raise NotImplementedError()


    plt.show()


if __name__ == "__main__":
    main()
