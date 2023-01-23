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
from kmedians import KMedians, l1_distances
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
    bestModel = "NOT IMPLEMENTED"
    bestModelLoss = 0
    for rep in range(reps):
        currModel = cls(X, k, plot=False, log=False)
        currModelLoss = currModel.loss(X)
        if bestModel == "NOT IMPLEMENTED":
            bestModel = currModel
            bestModelLoss = currModelLoss
        else:
            if (currModelLoss < bestModelLoss):
                bestModel = currModel
                bestModelLoss = currModelLoss
    
    return bestModel


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
    # A = np.array([[1, 2], [3, 4]])
    # B = np.array([[-1, 1], [0, 3]])


    # print(f"B looks like:\n {B}\n")
    # print(f"Their pairwise L1 distances is: \n{l1_distances(A, B)}")


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
    ## tips from https://towardsdatascience.com/how-to-split-a-dataset-into-training-and-testing-sets-b146b1649830
    n, d = X.shape
    split_separator = np.random.rand(n) <= 0.5
    train_X, train_y = X[split_separator], y[split_separator]
    validation_X, validation_y = X[~split_separator], y[~split_separator]

    lams = np.exp(np.linspace(-4, 4, num=15))
    sigmas = np.linspace(0, 8, num = 16)

    print(f"Searchspace for lambda is \n {lams}\n")
    print(f"Searchspace for sigma is \n {sigmas}\n")

    bestModel = "NOT IMPLEMENTED"
    bestError = 0
    for lam in lams:
        for sigma in sigmas:
            currModel = LeastSquaresRBFL2(train_X, train_y, lam=lam, sigma=sigma)
            validation_yhats = currModel.predict(validation_X)
            error_rate = np.mean((validation_yhats - validation_y) ** 2)
            if (bestModel == "NOT IMPLEMENTED"):
                bestError = error_rate
                bestModel = currModel
            else:
                if (error_rate < bestError):
                    bestModel = currModel
                    bestError = error_rate

    print(f"Out of the search, the best model had validation error of {bestError}")
    print(f"with lambda = {bestModel.lam} and sigma = {bestModel.sigma}")

    test_and_plot(bestModel, X, y, filename="leastsquares-rbfl2-selected.png")



@handle("7")
def q_7():
    X, y, X_test, y_test = load_dataset("multiData", "X", "y", "Xtest", "ytest")

    ovr_lr = OneVsRestClassifier(LogisticRegression, X, y)
    ovr_lr_y_hat = ovr_lr.predict(X_test)
    print(f"Logistic regression test error: {np.mean(ovr_lr_y_hat != y_test):.1%}")
    plot2Dclassifier(
        ovr_lr, X, y, X_test=X_test, y_test=y_test, k=5, filename="lr-preds.png"
    )

    softmax = SoftmaxClassifier(X, y)
    softmax_y_hat = softmax.predict(X_test)
    print(f"Softmax test error: {np.mean(softmax_y_hat != y_test):.1%}")
    plot2Dclassifier(
        softmax, X, y, X_test=X_test, y_test=y_test, k=5, filename="lr-preds-softmax.png"
    )

    # raise NotImplementedError()


    plt.show()


if __name__ == "__main__":
    main()
