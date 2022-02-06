#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, handle, main
from kmeans import Kmeans

@handle("1")
def basic_Kmeans_clusters():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    model.error(X, y, model.means)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("plots", "kmeans_basic_clustering_rand.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("2")
def clustering_min_error():
    X = load_dataset("clusterData.pkl")["X"]
    min = np.inf
    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)

        modelError = model.error(X,y,model.means)
        if modelError < min:
            min = modelError
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
            print(f"Min error updated: {modelError}")

            fname = Path("plots", "kmeans_minError_k=4.png")
            plt.savefig(fname)
            print(f"Figure saved as {fname}")


@handle("3")
def min_error_vs_k():
    X = load_dataset("clusterData.pkl")["X"]

    ks = range(1,11)
    knn_errors = np.zeros(10)

    for k in ks:

        minError = np.inf

        for i in range(50):
            model = Kmeans(k)
            model.fit(X)
            y = model.predict(X)
            modelError = model.error(X, y, model.means)

            if modelError < minError:
                minError = modelError

        knn_errors[ks.index(k)] = minError

    plt.plot(ks, knn_errors, label="min_error")
    plt.xlabel("Value of K")
    plt.ylabel("Minimum error")

    fname = Path("plots", "minErrorVsK.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")

if __name__ == "__main__":
    main()
