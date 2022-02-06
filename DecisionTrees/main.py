#!/usr/bin/env python
import os
from pathlib import Path
import time

# 3rd party libraries
# To make sure you have all of these (but they're all standard):
#    conda install numpy pandas matplotlib-base scipy scikit-learn
# or
#    pip install numpy pandas matplotlib scipy scikit-learn
# Annoyingly, Python's package manager names are not always the same
# as the import names (e.g. scikit-learn vs sklearn).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())


from utils import plot_classifier, mode, load_dataset, handle, main, run
from decision_stump import (
    DecisionStumpEquality,
    DecisionStumpErrorRate,
    DecisionStumpInfoGain,
)
from decision_tree import DecisionTree


@handle("1")
def basicDecisionStump_accuracy():
    # Load citiesSmall dataset
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # Evaluate decision stump
    model = DecisionStumpErrorRate()
    model.fit(X, y)
    y_pred = model.predict(X)

    error = np.mean(y_pred != y)
    print("Decision Stump with inequality rule error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model, X, y)

    fname = Path("plots", "SimpleError_DecisionBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("2")
def infoGainStump_accuracy():
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # Evaluate decision stump
    model = DecisionStumpInfoGain()
    model.fit(X, y)
    
    y_pred = model.predict(X)

    error = np.mean(y_pred != y)
    print("Decision Stump with info gain rule error: %.3f" % error)

    # PLOT RESULT
    plot_classifier(model, X, y)

    fname = Path("plots", "infoGain_decisionStumpBoundary.pdf")
    plt.savefig(fname)
    print("\nFigure saved as '%s'" % fname)


@handle("3")
def decisionTree():
    # Load citiesSmall dataset
    dataset = load_dataset("citiesSmall.pkl")
    X = dataset["X"]
    y = dataset["y"]

    # 2. Evaluate decision tree
    depth = 4
    model = DecisionTree(max_depth=depth, stump_class=DecisionStumpInfoGain)
    model.fit(X, y)
    y_pred = model.predict(X)
    error = np.mean(y_pred != y)

    print(f"Error for decision tree with depth 4: {error:.3f}")

    plot_classifier(model, X, y)

    plotname = "depth" + str(depth) + "DecisionTree_Boundary.pdf"

    fname = Path("plots", plotname)
    plt.savefig(fname)
    print(f"\nFigure saved as {fname}")

    def print_stump(stump):
        print(
            f"Splitting on feature {stump.j_best} at threshold {stump.t_best:f}. "
            f">: {stump.y_hat_yes}, <=: {stump.y_hat_no}"
        )

    print("Top:")
    print_stump(model.stump_model)
    print(">")
    print_stump(model.submodel_yes.stump_model)
    print("<=")
    print_stump(model.submodel_no.stump_model)


@handle("4")
def compareDiff():
    dataset = load_dataset("citiesSmall")
    X = dataset["X"]
    y = dataset["y"]
    print(f"n = {X.shape[0]}")

    depths = np.arange(1, 15)  # depths to try

    t = time.time()
    my_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        # model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpEquality)
        model = DecisionTree(max_depth=max_depth)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors[i] = np.mean(y_pred != y)
    print(
        f"Our decision tree with DecisionStumpErrorRate took {time.time() - t} seconds"
    )

    plt.plot(depths, my_tree_errors, label="errorrate")

    t = time.time()
    my_tree_errors_infogain = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTree(max_depth=max_depth, stump_class=DecisionStumpInfoGain)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors_infogain[i] = np.mean(y_pred != y)
    print(
        f"Our decision tree with DecisionStumpInfoGain took {time.time() - t} seconds"
    )

    plt.plot(depths, my_tree_errors_infogain, label="infogain")

    t = time.time()
    sklearn_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTreeClassifier(
            max_depth=max_depth, criterion="entropy", random_state=1
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        sklearn_tree_errors[i] = np.mean(y_pred != y)
    print(f"scikit-learn's decision tree took {time.time() - t} seconds")

    plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)

    plt.xlabel("Depth of tree")
    plt.ylabel("Classification error")
    plt.legend()
    fname = Path("plots", "different_tree_errors.pdf")
    plt.savefig(fname)

    # # plot the depth 15 sklearn classifier
    # model = DecisionTreeClassifier(max_depth=15, criterion="entropy", random_state=1)
    # model.fit(X, y)
    # plot_classifier(model, X, y)
    # fname = Path("..", "figs", "q6_5_decisionBoundary.pdf")
    # plt.savefig(fname)
    # print("\nFigure saved as '%s'" % fname)


if __name__ == "__main__":
    main()
