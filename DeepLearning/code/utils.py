import os, argparse
from pathlib import Path
import pickle, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats


def load_dataset(filename, *keys):
    if not filename.endswith(".pkl"):
        filename = filename + ".pkl"
    with open(Path("..", "data", filename), "rb") as (f):
        data = pickle.load(f)
        if not keys:
            return data
        return [data[k] for k in keys]

def load_dataset_adv(filename, *keys):
    pth = Path("..", "data", filename)

    if not pth.exists() and not pth.suffix:
        for suff in [".npz", ".pkl"]:
            alt = pth.with_suffix(suff)
            if alt.exists():
                pth = alt
                break

    data = np.load(pth, allow_pickle=True)
    if not keys:
        return dict(**data)
    return [data[k] for k in keys]

def test_and_plot(model, X, y, X_test=None, y_test=None, title=None, filename=None):
    yhat = model.predict(X)
    trainError = np.mean((yhat - y) ** 2)
    print("Training error = %.1f" % trainError)
    if X_test is not None:
        if y_test is not None:
            yhat = model.predict(X_test)
            testError = np.mean((yhat - y_test) ** 2)
            print("Test error     = %.1f" % testError)
    plt.figure()
    plt.plot(X, y, "b.")
    Xgrid = np.linspace(np.min(X), np.max(X), 1000)[:, None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, "g")
    if title is not None:
        plt.title(title)
    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def euclidean_dist_squared(X, X_test):
    """Computes the Euclidean distance between rows of 'X' and rows of 'X_test'

    Parameters
    ----------
    X : an N by D numpy array
    X_test: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows extremely-fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD registers and
    multiple cores (and faster matrix-multiplication algorithms), vector and matrix operations in
    Numpy will often be several times faster than if you implemented them yourself in a fast
    language like C. The following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in D, then
    element D[i,j] gives the squared Euclidean distance between training point
    i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """
    return (
        np.sum((X**2), axis=1)[:, None]
        + np.sum((X_test**2), axis=1)[None]
        - 2 * np.dot(X, X_test.T)
    )


def plot2Dclusters(X, y, w=None, filename=None):
    k = np.unique(y).size
    symbols = [
        "'s'",
        "'o'",
        "'v'",
        "'^'",
        "'x'",
        "'+'",
        "'*'",
        "'d'",
        "'<'",
        "'>'",
        "'p'",
    ]
    for c in range(k):
        colour = (0.75 * COLOURS[c][0], 0.75 * COLOURS[c][1], 0.75 * COLOURS[c][2])
        plt.scatter(
            (X[(y == c, 0)]), (X[(y == c, 1)]), marker=(symbols[c]), color=colour, s=10
        )
        if w is not None:
            plt.scatter(
                (w[(c, 0)]), (w[(c, 1)]), marker=(symbols[c]), color=(COLOURS[c]), s=100
            )

    if filename is not None:
        plt.savefig(f"../figs/{filename}", bbox_inches="tight", pad_inches=0.1)
        print(f"Plot saved as {filename}")


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
    questions = sorted(_funcs.keys())
    parser.add_argument(
        "questions",
        choices=(questions + ["all"]),
        nargs="+",
        help="A question ID to run, or 'all'.",
    )
    args = parser.parse_args()
    for q in args.questions:
        if q == "all":
            for q in sorted(_funcs.keys()):
                start = f"== {q} "
                print("\n" + start + "=" * (80 - len(start)))
                run(q)

        else:
            run(q)
