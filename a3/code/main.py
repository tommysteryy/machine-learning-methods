#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import pickle

import numpy as np
from scipy.special import betaln


# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    test_and_plot,
    load_dataset,
    main,
    handle,
    run,
)


def bernoulli_nll(theta, n_pos, n_total):
    return -n_pos * np.log(theta) - (n_total - n_pos) * np.log1p(-theta)
    # np.log1p(x) == np.log(1 + x)  except that log1p is more accurate for small x


def berns_nll(thetas, data):
    return sum(
        bernoulli_nll(theta, n_pos, n_total)
        for theta, (n_pos, n_total) in zip(thetas, data)
    )


@handle("eb-base")
def eb_base():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    n_groups = n.shape[0]

    theta = np.full(n_groups, 0.5)
    print(f"NLL for theta = 0.5: {berns_nll(theta, n_test) : .1f}")

    mle_thetas = n[:, 0] / n[:, 1]
    with np.errstate(all="ignore"):  # ignore numerical warnings about nans
        print(f"NLL for theta = MLE: {berns_nll(mle_thetas, n_test) : .1f}")


@handle("eb-map")
def eb_map():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



@handle("eb-bayes")
def eb_bayes():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



@handle("eb-pooled")
def eb_pooled():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



@handle("eb-max")
def eb_max():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



@handle("eb-newprior")
def eb_newprior():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



@handle("eb-newprior-sep")
def eb_newprior_sep():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    raise NotImplementedError()



################################################################################


def eval_models(models, ds_name="mnist35"):
    X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
    # bad design decisions here, sorry:
    #   for mnist35, y and ytest are both {0, 1}-valued; for mnist, they're one-hot
    # these are what TorchNeuralNetClassifier.fit expects
    # but it always returns an integer label
    if len(y.shape) == 2:
        ytest_int = np.argmax(ytest, axis=1)  # turn one-hot labels into integers
    else:
        ytest_int = ytest

    for model in models:
        model.fit(X, y)
        yhat = model.predict(Xtest)

        assert yhat.shape == ytest_int.shape
        yield np.mean(yhat != ytest_int)


def eval_model(model, ds_name="mnist35"):
    return next(eval_models([model], ds_name))


@handle("nns-35")
def nns():
    import neural_net as nn

    names, models = zip(
        *[
            # you might want to comment one out while fiddling
            ("manual", nn.NeuralNetClassifier([3])),
            ("torch", nn.TorchNeuralNetClassifier([3], device="cpu")),
            # might run faster if you change device= to use your GPU:
            #    "mps" if you have a recent Mac
            #    "cuda" on Linux/Windows if you have an appropriate GPU and PyTorch install
        ]
    )
    for name, err in zip(names, eval_models(models, "mnist35")):
        print(f"{name} test set error: {err:.1%}")


@handle("nns-10way")
def nns():
    import neural_net as nn

    names, models = zip(
        *[
            # you might want to comment one out while fiddling
            ("torch", nn.TorchNeuralNetClassifier([3], device="cpu")),
            ("cnn", nn.Convnet(device="cpu")),
        ]
    )
    for name, err in zip(names, eval_models(models, "mnist")):
        print(f"{name} test set error: {err:.1%}")


if __name__ == "__main__":
    main()
