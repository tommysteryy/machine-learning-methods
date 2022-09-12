#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LeastSquaresLossL2,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
import utils
from utils import load_dataset, load_trainval, load_and_split


# this just some Python scaffolding to conveniently run the functions below;
# don't worry about figuring out how it works if it's not obvious to you
_funcs = {}


def handle(number):
    def register(func):
        _funcs[number] = func
        return func

    return register


def run(question):
    if question not in _funcs:
        raise ValueError(f"unknown question {question}")
    return _funcs[question]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q", "--question", required=True, choices=sorted(_funcs.keys()) + ["all"]
    )
    args = parser.parse_args()
    if args.question == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(lr_model, X_train, y_train)
    utils.savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = KernelLogisticRegressionLossL2(0.01)
    optimizer = GradientDescentLineSearch()
    # kernel = PolynomialKernel(2)
    kernel = GaussianRBFKernel(0.5)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = utils.plot_classifier(klr_model, X_train, y_train)
    utils.savefig("logRegGaussian.png", fig)

@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    """YOUR CODE HERE FOR Q1.2"""
    for i in range(len(sigmas)):
        for j in range(len(lammys)):
            loss_fn = KernelLogisticRegressionLossL2(lammys[j])
            optimizer = GradientDescentLineSearch()
            kernel = GaussianRBFKernel(sigmas[i])
            model = KernelClassifier(loss_fn, optimizer, kernel)
            model.fit(X_train, y_train)

            train_err = np.mean(model.predict(X_train) != y_train)
            val_err = np.mean(model.predict(X_val) != y_val)

            train_errs[i,j] = train_err
            val_errs[i,j] = val_err

    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    utils.savefig("logRegRBF_grids.png", fig)


    best_train_err_index = np.unravel_index(np.argmin(train_errs, axis=None), train_errs.shape)
    best_train_sigma = sigmas[best_train_err_index[0]]
    best_train_lammy = lammys[best_train_err_index[1]]
    best_train_error = train_errs[best_train_err_index]

    print(f"train_errs: {train_errs}\n")
    print(f"Best train error index: {best_train_err_index} \n")
    print(f"Best train_err sigma: {best_train_sigma}\n")
    print(f"Best train_err lambda: {best_train_lammy}\n")
    print(f"Best training error calculated: {best_train_error}")

    best_val_err_index = np.unravel_index(np.argmin(val_errs, axis=None), val_errs.shape)
    best_val_sigma = sigmas[best_val_err_index[0]]
    best_val_lammy = lammys[best_val_err_index[1]]
    best_val_error = val_errs[best_val_err_index]

    print(f"val_errs: {val_errs}\n")
    print(f"Best val error index: {best_val_err_index} \n")
    print(f"Best val_err sigma: {best_val_sigma}\n")
    print(f"Best val_err lambda: {best_val_lammy}\n")
    print(f"Best validation error calculated: {best_val_error}")

    ## plot with best params
    best_hypers = [(best_train_sigma, best_train_lammy), (best_val_sigma, best_val_lammy)]
    count = 1
    for s,l in best_hypers:
        loss_fn = KernelLogisticRegressionLossL2(l)
        optimizer = GradientDescentLineSearch()
        kernel = GaussianRBFKernel(s)
        model = KernelClassifier(loss_fn, optimizer, kernel)
        model.fit(X_train, y_train)

        fig = utils.plot_classifier(model, X_train, y_train)

        utils.savefig(f"decisionBoundary{count}", fig)

        count += 1




@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = utils.standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # # Matrix plot
    # fig, ax = plt.subplots()
    # ax.imshow(X_train_standardized)
    # utils.savefig("animals_matrix.png", fig)

    # # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    # ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    # for i in random_is:
    #     xy = X_train_standardized[i, [j1, j2]]
    #     ax.annotate(animal_names[i], xy=xy)
    # utils.savefig("animals_random.png", fig)
    k = 2
    # pca_encoder = PCAEncoder(k)
    # pca_encoder.fit(X_train)
    # Z_train = X_train_standardized @ pca_encoder.W.T
    #
    # ax.scatter(Z_train[:,0], Z_train[:,1])
    #
    # for i in random_is:
    #     xy = Z_train[i]
    #     ax.annotate(animal_names[i], xy=xy)
    # utils.savefig("animals_PCA.png", fig)
    #
    # for k in range(10):
    #     pca_encoder = PCAEncoder(k)
    #     pca_encoder.fit(X_train)
    #     Z_train = X_train_standardized @ pca_encoder.W.T
    #
    #     variance_projected = np.linalg.norm(Z_train @ pca_encoder.W - X_train_standardized) ** 2
    #     variance_total = np.linalg.norm(X_train_standardized) ** 2
    #
    #     variance_explained_by = 1 - variance_projected / variance_total
    #     print(f"Variance explained by {k} PCs: {variance_explained_by:.3%}")
    #
    #     if variance_explained_by > 0.5:
    #         break

    n,d = X_train.shape

    pca_encoder = PCAEncoder(1)
    pca_encoder.fit(X_train)
    fake_W = np.zeros((2,d))
    fake_W[0] =  pca_encoder.W.T[0]
    fake_W[1] = pca_encoder.W.T[0] * 2
    Z_train = X_train_standardized @ pca_encoder.W.T

    variance_projected = np.linalg.norm(Z_train @ pca_encoder.W - X_train_standardized) ** 2
    variance_total = np.linalg.norm(X_train_standardized) ** 2

    variance_explained_by = 1 - variance_projected / variance_total
    print(f"Variance explained by 1 PCs: {variance_explained_by:.3%}")

    Z_train_fake = X_train_standardized @ fake_W.T
    variance_projected_fake = np.linalg.norm(Z_train_fake @ fake_W - X_train_standardized) ** 2

    variance_explained_by_fake = 1 - variance_projected_fake / variance_total
    print(f"Variance explained by 2 duplicate PCs: {variance_explained_by_fake:.3%}")


@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    utils.savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    # batch_size = 1
    # batch_size = 10
    # batch_size = 100
    batchsizes = [1, 10, 100]
    for bs in batchsizes:
        constantLRG = ConstantLR(0.0003)
        loss_fn = LeastSquaresLoss()
        optimizer = GradientDescent()
        stochastic_optimizer = StochasticGradient(optimizer, constantLRG, bs, max_evals=10)
        model = LinearModel(loss_fn, stochastic_optimizer, check_correctness=False)
        model.fit(X_train, y_train)

        print(f"Training MSE for batch size = {bs}: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
        print(f"Validation MSE for batch size = {bs}: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = utils.standardize_cols(X_train_orig)
    X_val, _, _ = utils.standardize_cols(X_val_orig, mu, sigma)

    c = 0.1

    constantLRG = ConstantLR(c)
    inverseLRG = InverseLR(c)
    invSquaredLRG = InverseSquaredLR(c)
    invSqrtLRG = InverseSqrtLR(c)

    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescent()

    fig, ax = plt.subplots()
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")

    stochastic_optimizer_constant = StochasticGradient(optimizer, constantLRG, 10, max_evals=75)
    model_constant = LinearModel(loss_fn, stochastic_optimizer_constant, check_correctness=False)
    model_constant.fit(X_train, y_train)

    stochastic_optimizer_inv = StochasticGradient(optimizer, inverseLRG, 10, max_evals=75)
    model_inv = LinearModel(loss_fn, stochastic_optimizer_inv, check_correctness=False)
    model_inv.fit(X_train, y_train)

    stochastic_optimizer_invsquared = StochasticGradient(optimizer, invSquaredLRG, 10, max_evals=75)
    model_invsquared = LinearModel(loss_fn, stochastic_optimizer_invsquared, check_correctness=False)
    model_invsquared.fit(X_train, y_train)

    stochastic_optimizer_invsqrt = StochasticGradient(optimizer, invSqrtLRG, 10, max_evals=75)
    model_invsqrt = LinearModel(loss_fn, stochastic_optimizer_invsqrt, check_correctness=False)
    model_invsqrt.fit(X_train, y_train)

    ax.plot(model_constant.fs, marker="o", label="constant")
    ax.plot(model_inv.fs, marker="s", label="inverse")
    ax.plot(model_invsquared.fs, marker="^", label="inv_squared")
    ax.plot(model_invsqrt.fs, marker="x", label="inv_sqrt")
    plt.legend()
    utils.savefig("gd_diff_LRG_curve.png", fig)

if __name__ == "__main__":
    main()
