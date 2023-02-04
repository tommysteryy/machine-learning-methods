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
from utils import (
    test_and_plot,
    load_dataset,
    main,
    handle,
    run,
)
from naive_bayes import NaiveNaiveBayes, NaiveBayes, VQNB

from logistic_regression import LogisticRegression
from neural_net import NeuralNetRegressor, NeuralNetClassifier


def bernoulli_sample(n_sample, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    raise NotImplementedError()



@handle("1f")
def q_1f():
    max_n = 100_000
    theta = 0.17
    n_repeats = 3

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Monte Carlo approximation")
    ax.set_xlim(1, max_n)
    ax.set_ylim(-2, 1)

    raise NotImplementedError()


    fn = "../figs/bernoulli_game.pdf"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {fn}")


def eval_models(models, ds_name="mnist35_binary"):
    X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
    for model in models:
        model.fit(X, y)
        yhat = model.predict(Xtest)
        yield np.mean(yhat != ytest)


def eval_model(model, ds_name="mnist35_binary"):
    return next(eval_models([model], ds_name))


@handle("3")
def q_3():
    err = eval_model(NaiveNaiveBayes())
    print(f"Test set error: {err:.1%}")


@handle("3c")
def q_3c():
    laps = [1e-9, 0.5, 1, 2.5, 5, 10, 50]
    errs = eval_models([NaiveBayes(laplace_smooth=lap) for lap in laps])
    print("NaiveBayes test set errors:")
    for lap, err in zip(laps, errs):
        print(f"  smooth {lap:>4.1f}:  {err:.1%}")




@handle("3-logistic")
def q3_logistic():
    err = eval_model(LogisticRegression())
    print(f"Test set error: {err:.1%}")


@handle("3-vqnb")  # question 3 (e), (g)
def q3_vqnb():
    ks = [2, 3, 4, 5]
    models = [VQNB(k=k) for k in ks]
    print("VQNB test set errors:")
    for k, err in zip(ks, eval_models(models)):
        print(f"  k = {k}:  {err:.1%}")

    model = models[-1]
    fig, axes = plt.subplots(
        2, 5, figsize=(20, 8), sharex=True, sharey=True, constrained_layout=True
    )
    for y in range(2):
        for c in range(5):
            ps = np.zeros(784)  # get the probabilities from your model
            axes[y][c].imshow(ps.reshape((28, 28)), "gray")
    fig.savefig("../figs/vqnb_probs.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Plots in ../figs/vqnb_probs.pdf")


@handle("4.2")
def q_4_2():
    X, y = load_dataset("basisData", "X", "y")

    model = NeuralNetRegressor([5])
    model.fit(X, y)
    yhat = model.predict(X)
    print("Training error: ", np.mean((yhat - y) ** 2))
    print("Figure in ../figs/regression-net.pdf")




@handle("4.3")
def q_4_3():
    X, y, Xtest, ytest = load_dataset("mnist35", "X", "y", "Xtest", "ytest")

    model = NeuralNetClassifier([5])
    model.fit(X, y)
    yhat = model.predict(Xtest)
    print(f"Test set error: {np.mean(yhat != ytest):.1%}")



if __name__ == "__main__":
    main()
