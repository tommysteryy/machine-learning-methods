#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Mar 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    load_dataset,
    main,
    handle
)

from markov_chain import MarkovChain

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



@handle("mc-mostlikely-sequence")
def mc_mostlikely_sequence():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    # print(p1)

    for time in [50, 100]:
        print(f"Decoding to time {time}: {model.mode(time)}")


@handle("mc-conditional-exact")
def mc_conditional_mc():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)
    print(pt)
    cond_probs = []
    total = 0
    for i in range(p1.size):
        prob = model.mc_conditionals(3, i, 6, 5)
        print(f"p(x_3 = {i}, x_6 = 5) = {prob}")

        total += prob
        cond_probs.append(prob)
    
    print(cond_probs * 1/total)


@handle("mc-conditional-mc")
def mc_conditional_exact():
    p1, pt = load_dataset("gradChain", "p1", "pt")
    model = MarkovChain(p1, pt)

    n_samples = 10000
    time = 6
    samples = model.sample(n_samples, time)

    accepted = 0
    counts_c = np.zeros(7)

    for sample in samples:
        x_6 = sample[5]
        x_3 = sample[2]

        if (x_6 == 5):
            accepted += 1
            counts_c[x_3] = counts_c[x_3] + 1

    rejection_rate = 1 - accepted / n_samples
    print(f"rejection rate = {rejection_rate}")
    print(f"Accepted: {accepted}")
    print(f"Estimated probabilities = {counts_c / accepted}")


def bernoulli_sample(n_sample, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sample = rng.random(n_sample)
    return (sample < theta).astype(int)


@handle("mcmc-bernoulli")
def mcmc_bernoulli():
    max_n = 100_000
    theta = 0.17
    n_repeats = 3

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Monte Carlo approximation")
    ax.set_xlim(1, max_n)
    ax.set_ylim(-2, 1)

    def calc(sample):
        ## assumes sample is an array of length 1
        if sample[0] == 1:
            return -5
        else:
            return 1

    for run in range(n_repeats):
        running_sum = 0
        means = []
        for i in range(max_n):
            running_sum += calc(bernoulli_sample(1, theta))
            means.append(running_sum / (i+1))
        ax.plot(means)
    
    ## Expected value
    ax.axhline(y = -0.02)

    fn = "../figs/bernoulli_game.pdf"
    fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved in {fn}")

if __name__ == "__main__":
    main()
