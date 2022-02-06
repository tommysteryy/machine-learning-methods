#!/usr/bin/env python
import os
from pathlib import Path

import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, handle, main

from naive_bayes import NaiveBayes, NaiveBayesLaplace

@handle("1")
def training_and_validation_error():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("2")
def q():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model_nb = NaiveBayes(num_classes=4)
    model_nb.fit(X, y)

    y_hat_nb = model_nb.predict(X)
    err_train = np.mean(y_hat_nb != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat_nb = model_nb.predict(X_valid)
    err_valid = np.mean(y_hat_nb != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

    model_ls = NaiveBayesLaplace(num_classes=4, beta = 1)
    model_ls.fit(X,y)

    y_hat_ls = model_ls.predict(X)
    err_train = np.mean(y_hat_ls != y)
    print(f"Naive Bayes (Laplace) training error: {err_train:.3f}")

    y_hat_ls = model_ls.predict(X_valid)
    err_valid = np.mean(y_hat_ls != y_valid)
    print(f"Naive Bayes (Laplace) validation error: {err_valid:.3f}")


if __name__ == "__main__":
    main()
