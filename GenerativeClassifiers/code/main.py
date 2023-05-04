#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Mar 2023)
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
from utils import (
    load_dataset,
    main,
    handle
)

from knn import KNN
from generative_classifiers import GDA, TDA


def eval_models(models, ds_name):
    X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
    for model in models:
        model.fit(X, y)
        yhat = model.predict(Xtest)
        yield np.mean(yhat != ytest)


def eval_model(model, ds_name):
    return next(eval_models([model], ds_name))


@handle("gda")
def gda():
    model = KNN(1)
    print(f"{model.k}-NN test error: {eval_model(model, 'gaussNoise'):.1%}")
    
    model = GDA()
    print(f"GDA  test error: {eval_model(model, 'gaussNoise'):.1%}")


@handle("tda")
def tda():
    model = TDA()
    print(f"TDA  test error: {eval_model(model, 'gaussNoise'):.1%}")


if __name__ == "__main__":
    main()
