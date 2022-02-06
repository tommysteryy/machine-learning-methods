#!/usr/bin/env python
import os
from pathlib import Path

import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths

os.chdir(Path(__file__).parent.resolve())

# our code
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from utils import load_dataset, handle, main

from random_tree import RandomForest, RandomTree

@handle("1")
def randomForestPerformance():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Random Forest info gain")
    evaluate_model(RandomForest(50, np.inf))
    #
    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

    print("Random Tree info gain")
    evaluate_model(RandomTree(np.inf))

if __name__ == "__main__":
    main()
