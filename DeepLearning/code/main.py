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
    load_dataset,
    load_dataset_adv,
    main,
    handle
)
from neural_net import NeuralNetRegressor, NeuralNetClassifier




@handle("4.2")
def q_4_2():
    X, y = load_dataset("basisData", "X", "y")

    model = NeuralNetRegressor([75, 50, 25, 10, 20, 45, 50, 75],
                            #  These parameters are changed within the NeuralNetRegressor class
                            #    batch_size = 100,
                            #    activation = "sigmoid"
                               )
    model.fit(X, y)
    yhat = model.predict(X)
    print("Training error: ", np.mean((yhat - y) ** 2))
    print("Figure in ../figs/regression-net.pdf")




@handle("4.3")
def q_4_3():
    X, y, Xtest, ytest = load_dataset("mnist35", "X", "y", "Xtest", "ytest")

    model = NeuralNetClassifier(hidden_layer_sizes=[50, 25, 10, 20, 45], 
                                batch_size = 100, 
                                init_scale=0.5, 
                                learning_rate = 0.0005)
    model.fit(X, y)
    yhat = model.predict(Xtest)
    print(f"Test set error: {np.mean(yhat != ytest):.1%}")


def eval_models(models, ds_name="mnist35"):
    X, y, Xtest, ytest = load_dataset_adv(ds_name, "X", "y", "Xtest", "ytest")
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

        print(model)
    
        for param in model.parameters():
            print(param.numel())
        # print(total_params)

        yield np.mean(yhat != ytest_int)

    

def eval_model(model, ds_name="mnist35"):
    return next(eval_models([model], ds_name))


@handle("nns-10way")
def nns():
    import neural_net_adv as nn_adv

    names, models = zip(
        *[
            ("torch", nn_adv.TorchNeuralNetClassifier([64, 48, 48, 32, 32],
                                                  learning_rate= 0.0000125,
                                                  init_scale = 0.15, 
                                                  batch_size = 100,
                                                  weight_decay = 0.0001,
                                                  max_iter = 25000,
                                                  device="cpu")),
            ("torch", nn_adv.Convnet(device="cpu",
                                                  learning_rate= 0.0000125,
                                                  init_scale = 0.01, 
                                                  batch_size = 10,
                                                  weight_decay = 0.005,
                                                  max_iter = 20000,
                                                  )),
        ]
    )
    for name, err in zip(names, eval_models(models, "mnist")):
        print(f"{name} test set error: {err:.1%}")



if __name__ == "__main__":
    main()
