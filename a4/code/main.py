#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Mar 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle


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

from knn import KNN
from generative_classifiers import GDA, TDA
from markov_chain import MarkovChain
from bayesian_logistic_regression import BayesianLogisticRegression


def eval_models(models, ds_name):
    X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
    # print(X)
    # print(X.shape)
    # print(y)
    # print(y.shape)
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
    # X, y, Xtest, ytest = load_dataset("gaussNoise", "X", "y", "Xtest", "ytest")
    
    model = GDA()
    # print(X[:20])
    # print(y[:20])
    # print("==============")
    # # X_20 = X[:20]
    # # y_20 = y[:20]
    
    # print(X[y == 8][:5])
    # print(y[y == 8][:5])
    # model.fit(X, y)
    # # print(model.nll(X))
    # print(model.predict(Xtest)[:20])
    # print(ytest[:20])
    print(f"GDA  test error: {eval_model(model, 'gaussNoise'):.1%}")


@handle("tda")
def tda():
    model = TDA()
    print(f"TDA  test error: {eval_model(model, 'gaussNoise'):.1%}")




# END SOLUTION

################################################################################


@handle("mc-sample")
def mc_sample():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    n_samples = 10000
    time = 50
    samples = model.sample(n_samples, time)
    print(samples)
    est = np.bincount(samples[:, time - 1]) / n_samples
    print(f"Empirical dist, time {time}: {est.round(3)}")


@handle("mc-marginals")
def mc_marginals():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    # print(p1)
    # print(pt)

    n_samples = 10000
    time = 50
    samples = model.sample(n_samples, time)
    # print(samples)
    est = np.bincount(samples[:, time - 1]) / n_samples
    print(f"Empirical dist, time {time}: {est.round(3)}")

    marginals = model.marginals(time)
    print(marginals)
    print(marginals.argmax(axis=1))

    # raise NotImplementedError()



@handle("mc-mostlikely-marginals")
def mc_mostlikely():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    raise NotImplementedError()



@handle("mc-mostlikely-sequence")
def mc_mostlikely_sequence():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    # print(p1)

    for time in [50, 100]:
        print(f"Decoding to time {time}: {model.mode(time)}")


@handle("mc-gradschool-50")
def mc_gradschool_50():
    p1, pt = load_dataset("gradChain", "p1", "pt")

    # start_state = 2
    # start_time = 1
    # end = 10

    # def cond_prob(start_state, start_time, end_time):
    #     """
    #     Return the p1.size length vector with the conditional probabilities:
    #         p(x_end | x_start_time = start_state)
    #     CONSTAINTS: 
    #     - end must be at least 1 away from start_time
    #     - everything must be valid
    #     """
    #     if (end_time == 1):
    #         return pt[start_state]
    #     else:
    #         return pt.T @ cond_prob(start_state, start_time, end_time -1) 
    
    # print(cond_prob(start_state, start_time, end))
    print(p1)
    print(pt)




@handle("mc-conditional-mc")
def mc_conditional_mc():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    print(model.mc_conditionals(3, 6, 0))



@handle("mc-conditional-exact")
def mc_conditional_exact():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    raise NotImplementedError()



################################################################################


@handle("mcmc-blogreg")
def mcmc_blogreg():
    X, y = load_dataset("twoThrees", "X", "y")

    model = BayesianLogisticRegression()
    samples = model.sample_weights(X, y)
    model.plot_weights(samples, figname="blogreg")


if __name__ == "__main__":
    main()
