#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import betaln


# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    load_dataset,
    main,
    handle
)
from naive_bayes import NaiveNaiveBayes, NaiveBayes, VQNB



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
    # X, y, Xtest, ytest = load_dataset("mnist35_binary", "X", "y", "Xtest", "ytest")
    # err = eval_model(NaiveBayes())
    # print(f"Test set error: {err:.1%}")


@handle("3-vqnb") 
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
            # ps = np.zeros(784)  # get the probabilities from your model
            ps = model.p_xj_yz[:, y, c]
            axes[y][c].imshow(ps.reshape((28, 28)), "gray")
    fig.savefig("../figs/vqnb_probs.pdf", bbox_inches="tight", pad_inches=0.1)
    print("Plots in ../figs/vqnb_probs.pdf")


def bernoulli_nll(theta, n_pos, n_total):
    return -n_pos * np.log(theta) - (n_total - n_pos) * np.log1p(-theta)
    # np.log1p(x) == np.log(1 + x)  except that log1p is more accurate for small x


def berns_nll(thetas, data):
    return sum(
        bernoulli_nll(theta, n_pos, n_total)
        for theta, (n_pos, n_total) in zip(thetas, data)
    )

@handle("eb-bayes")
def eb_bayes():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    alpha = 2; beta = 2
    n_groups = n.shape[0]

    nll_post_predictive_prob = 0

    for group in range(n_groups):
        n1 = n[group, 0]
        n0 = n[group, 1] - n1

        alpha_tilde = n1 + alpha
        beta_tilde =  n0 + beta

        n1_tilde = n_test[group, 0]
        n0_tilde = n_test[group, 1] - n1_tilde

        nll_post_predictive_prob += -1*(betaln(n1_tilde + alpha_tilde, n0_tilde + beta_tilde) - betaln(alpha_tilde, beta_tilde))

    print(f"NLL with separate posterior predictive probability: { nll_post_predictive_prob : .1f}")

    ## p(xhat = 1 | X)
    



@handle("eb-pooled")
def eb_pooled():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    n_test_groups = n_test.shape[0]
    alpha = 2; beta = 2
    # alpha = 0.9; beta = 651

    total_sick = sum(n[:, 0])
    total = sum(n[:, 1])
    total_notsick = total - total_sick

    total_sick_test = sum(n_test[:, 0])
    total_test = sum(n_test[:, 1])
    total_notsick_test = total_test - total_sick_test

    thetas_MLE = np.full(n_test_groups, total_sick / total)
    thetas_MAP = np.full(n_test_groups, (total_sick + alpha -1 ) / 
                         ((total_sick + alpha -1) + (total + beta - 1)))
    posterior_pred = -1*(betaln(alpha + total_sick + total_sick_test, 
                                beta + total_notsick + total_notsick_test) - betaln(alpha + total_sick, 
                                                                                    beta + total_notsick_test))
    posterior_pred_new = (betaln(alpha + total_sick + 1, 
                                 beta + total_notsick + 0) - betaln(alpha + total_sick, beta + total_notsick_test))


    print(f"NLL with MLE: { berns_nll(thetas_MLE, n_test): .1f}")
    print(f"NLL with MAP: {berns_nll(thetas_MAP, n_test): .1f}")
    print(f"Posterior Predictive: {posterior_pred: .1f}")
    print(f"Posterior Predictive New: {posterior_pred_new: .1f}")

if __name__ == "__main__":
    main()
