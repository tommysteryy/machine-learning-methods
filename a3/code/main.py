#!/usr/bin/env python
# Written by Alan Milligan and Danica Sutherland (Jan 2023)
# Based on CPSC 540 Julia code by Mark Schmidt
# and some CPSC 340 Python code by Mike Gelbart and Nam Hee Kim, among others

import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn

import numpy as np
from scipy.special import betaln


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


def bernoulli_nll(theta, n_pos, n_total):
    return -n_pos * np.log(theta) - (n_total - n_pos) * np.log1p(-theta)
    # np.log1p(x) == np.log(1 + x)  except that log1p is more accurate for small x


def berns_nll(thetas, data):
    return sum(
        bernoulli_nll(theta, n_pos, n_total)
        for theta, (n_pos, n_total) in zip(thetas, data)
    )

def map_betaprior_nll(alpha, beta, thetas, data):
    return sum(
        bernoulli_nll(theta, n_pos + alpha - 1, n_total + beta -1)
        for theta, (n_pos, n_total) in zip(thetas, data)
    )


@handle("eb-base")
def eb_base():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    n_groups = n.shape[0]

    print(n)
    print(n_test)

    theta = np.full(n_groups, 0.5)
    print(f"NLL for theta = 0.5: {berns_nll(theta, n_test) : .1f}")

    mle_thetas = n[:, 0] / n[:, 1]
    with np.errstate(all="ignore"):  # ignore numerical warnings about nans
        print(f"NLL for theta = MLE: {berns_nll(mle_thetas, n_test) : .1f}")


@handle("eb-map")
def eb_map():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    # alpha = 2; beta = 2
    alpha = 0.9; beta = 651

    map_thetas = (n[:, 0] + alpha -1 ) / ((n[:, 0] + alpha -1 )+(n[:, 1] + beta - 1))

    ## this is the case that we're using posterior distribution
    # print(f"NLL with beta Prior of alpha = {alpha}, beta = {beta}: {map_betaprior_nll(alpha, beta, map_thetas, n_test) : .1f}")
    print(n)
    print(map_thetas)
    ## But you're supposed to just use the likelihood - the posterior is only for finding the optimal theta.
    print(f"NLL with beta Prior of alpha = {alpha}, beta = {beta}: {berns_nll(map_thetas, n_test) : .1f}")



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
    # alpha = 2; beta = 2
    alpha = 0.9; beta = 651

    total_sick = sum(n[:, 0])
    total = sum(n[:, 1])
    total_notsick = total - total_sick

    total_sick_test = sum(n_test[:, 0])
    total_test = sum(n_test[:, 1])
    total_notsick_test = total_test - total_sick_test

    thetas_MLE = np.full(n_test_groups, total_sick / total)
    thetas_MAP = np.full(n_test_groups, (total_sick + alpha -1 ) / ((total_sick + alpha -1 )+(total + beta - 1)))
    posterior_pred = -1*(betaln(alpha + total_sick + total_sick_test, beta + total_notsick + total_notsick_test) - betaln(alpha + total_sick, beta + total_notsick_test))
    posterior_pred_new = (betaln(alpha + total_sick + 1, beta + total_notsick + 0) - betaln(alpha + total_sick, beta + total_notsick_test))


    print(f"NLL with MLE: { berns_nll(thetas_MLE, n_test): .1f}")
    print(f"NLL with MAP: {berns_nll(thetas_MAP, n_test): .1f}")
    print(f"Posterior Predictive: {posterior_pred: .1f}")
    print(f"Posterior Predictive New: {posterior_pred_new: .1f}")







@handle("eb-max")
def eb_max():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    def log_marginal_LL(k, m, n1, n0):
        return betaln(m*k + n1, (k - m*k) + n0) - betaln(m*k, (k - m*k))

    n1 = sum(n[:, 0]); n0 = sum(n[:, 1]) - n1
    m = n1 / (n0 + n1)

    print(f"m = {m}")

    ks = range(1, 10000000, 10)
    nlls = []

    # print(nll)

    # for k_ in ks:
    #     nll = 
    #     nlls.append(nll)
    print(np.exp(log_marginal_LL(10000000, m, n1, n0)))

    # plt.plot(ks, nlls)
    # plt.xlabel("k")
    # plt.ylabel("Log Marginal Likelihood with m = MLE")
    # # plt.show()
    # plt.savefig("../figs/eb-max-plot.png")

    



@handle("eb-newprior")
def eb_newprior():
    n, n_test = load_dataset("cancerData", "n", "nTest")

    def ln_regularizer(m, k):
        return np.log(m**(-0.99) *(1-m)**8.9) + np.log(1/((1+k) ** 2))

    def log_marginal_LL(k, m, n1, n0):
        return  betaln(m*k + n1, (k - m*k) + n0) - betaln(m*k, (k - m*k)) + ln_regularizer(m, k)

    n1 = sum(n[:, 0]); n0 = sum(n[:, 1])
    
    alphas = np.linspace(0.1, 9.9, 99)
    betas = np.linspace(0.1, 9.9, 99)

    best_alpha = 0
    best_beta = 0
    best_ll_overall = -1000000

    for alpha in alphas:
        
        for beta in betas:
            k = alpha + beta
            m = alpha / k
            ll = log_marginal_LL(k, m, n1, n0)
            
            if ll > best_ll_overall:
                best_alpha = alpha
                best_beta = beta
                best_ll_overall = ll

    print(f"The best LL was achieved by a = {best_alpha}, b = {best_beta}, a LL of {best_ll_overall}")



@handle("eb-newprior-sep")
def eb_newprior_sep():
    n, n_test = load_dataset("cancerData", "n", "nTest")
    n_groups = n.shape[0]

    def ln_regularizer(alpha, beta):
        k = alpha + beta
        m = alpha / k

        return np.log(m**(-0.99) *(1-m)**8.9) + np.log(1/((1+k) ** 2))

    def log_marginal_LL_per_group(alpha, beta, n1, n0):
        """
        This is for a per-group calculation only
        """
        return betaln(alpha + n1, beta + n0) - betaln(alpha, beta)
    
    def LL_and_ln_regularizer(alpha, beta):
        log_marginal_likelihood = 0
        for group in range(n_groups):
            n1 = n[group, 0]
            n0 = n[group, 1] - n1
            log_marginal_likelihood += log_marginal_LL_per_group(alpha, beta, n1, n0)
        
        return log_marginal_likelihood + ln_regularizer(alpha, beta)
    
    def posterior_predictve(alpha, beta):

        post_predictive_prob = 0

        for group in range(n_groups):
            n1 = n[group, 0]
            n0 = n[group, 1] - n1

            alpha_tilde = n1 + alpha
            beta_tilde =  n0 + beta

            n1_tilde = n_test[group, 0]
            n0_tilde = n_test[group, 1] - n1_tilde

            post_predictive_prob += (betaln(n1_tilde + alpha_tilde, n0_tilde + beta_tilde) - betaln(alpha_tilde, beta_tilde))
        
        return post_predictive_prob
            

    alphas = np.linspace(0.1, 9.9, 99)
    betas = np.linspace(100, 1000)

    best_alpha = 0
    best_beta = 0
    best_ll_overall = -100000000

    for alpha in alphas:
        
        for beta in betas:
            ll = LL_and_ln_regularizer(alpha, beta)
            
            if ll > best_ll_overall:
                best_alpha = alpha
                best_beta = beta
                best_ll_overall = ll

    print(f"The best LL was achieved by a = {best_alpha}, b = {best_beta}, a LL of {best_ll_overall}")

    print(f"The best alpha, beta produced posterior-predictive probability of {posterior_predictve(0.9, 651)}")



################################################################################

def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    

def eval_models(models, ds_name="mnist35"):
    X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
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


@handle("nns-35")
def nns():
    import neural_net as nn
    from prettytable import PrettyTable

    names, models = zip(
        *[
            # you might want to comment one out while fiddling
            # ("manual", nn.NeuralNetClassifier([3])),
            ("torch", nn.TorchNeuralNetClassifier([3], 
                                                  
                                                  device="cpu")),
            # might run faster if you change device= to use your GPU:
            #    "mps" if you have a recent Mac
            #    "cuda" on Linux/Windows if you have an appropriate GPU and PyTorch install
        ]
    )
    for name, err in zip(names, eval_models(models)):
        print(f"{name} test set error: {err:.1%}")

    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad: continue
    #         params = parameter.numel()
    #         table.add_row([name, params])
    #         total_params+=params
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params
    
    # count_parameters(net)    
    
    # for param in models[0].parameters():
    #     print(f"param = {param}")
    #     print(f"Shape = {param.shape}")

    # layer_sizes = [784] + list([3]) + [1]
    # for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
    #     print(f"in_size = {in_size}; out_size = {out_size}")


@handle("nns-10way")
def nns():
    import neural_net as nn
    from prettytable import PrettyTable

    names, models = zip(
        *[
            # you might want to comment one out while fiddling
            # ("torch", nn.TorchNeuralNetClassifier([64, 48, 48, 32, 32],
            #                                     #   learning_rate= 0.0000125,
            #                                       init_scale = 0.15, 
            #                                       batch_size = 100,
            #                                       weight_decay = 0.0001,
            #                                       max_iter = 25000,
            #                                       device="cpu")),
            # ("cnn", nn.Convnet()),
            ("torch", nn.Convnet(device="cpu",
                                                #   learning_rate= 0.0000125,
                                                  init_scale = 0.01, 
                                                  batch_size = 10,
                                                  weight_decay = 0.005,
                                                  max_iter = 20000,
                                                  )),
        ]
    )
    for name, err in zip(names, eval_models(models, "mnist")):
        print(f"{name} test set error: {err:.1%}")

@handle("test-nn")
def testnn():
    # loss = nn.CrossEntropyLoss()
    # ## class probabilities
    # input = torch.randn(3, 5, requires_grad=True)
    # print(input)
    # target = torch.randn(3, 5).softmax(dim=1)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # output.backward()
    # # loss = nn.CrossEntropyLoss()   

    # ## class labels
    # input = torch.randn(3, 5, requires_grad=True)
    # print(input)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # output.backward()
    c = nn.Conv2d(1,128, kernel_size=(4,4))
    print(c.weight.shape)
    print(c.weight)


if __name__ == "__main__":
    main()
