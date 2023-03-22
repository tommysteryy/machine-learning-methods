import numpy as np
import torch
from torch import nn
import utils


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        W = np.reshape(
            weights_flat[counter : counter + W_size],
            (layer_sizes[i + 1], layer_sizes[i]),
        )
        counter += W_size
        weights.append(W)

    return weights


class NeuralNetClassifier:
    def __init__(
        self,
        hidden_layer_sizes,
        X=None,
        y=None,
        max_iter=10_000,
        learning_rate=0.0001,
        init_scale=1,
        batch_size=1,
        weight_decay=0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        if X is not None and y is not None:
            self.fit(X, y)

    def center(self, y):
        return 2 * y - 1

    def uncenter(self, yhat):
        return (yhat + 1) / 2

    def loss_and_grad(self, weights_flat, X, y):
        y = self.center(y)

        weights = unflatten_weights(weights_flat, self.layer_sizes)
        activations = [X]
        for W in weights:
            Z = X @ W.T
            X = np.tanh(Z)
            activations.append(X)

        yhat = Z
        f = 0.5 * np.sum((yhat - y) ** 2)
        grad = yhat - y
        grad_W = grad.T @ activations[-2]
        g = [grad_W]
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W = weights[i]
            grad = grad @ W
            grad = grad * (1 - activations[i] ** 2)
            grad_W = grad.T @ activations[i - 1]
            g = [grad_W] + g

        g = flatten_weights(g)
        return (f, g)

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = self.center(y)

        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        weights = list()
        for i in range(len(self.layer_sizes) - 1):
            W = self.init_scale * np.random.randn(
                self.layer_sizes[i + 1], self.layer_sizes[i]
            )
            weights.append(W)

        weights_flat = flatten_weights(weights)
        for i in range(self.max_iter):
            subset = np.random.choice((X.shape[0]), size=self.batch_size, replace=False)
            loss, step_gradient = self.loss_and_grad(weights_flat, X[subset], y[subset])
            weights_flat = weights_flat - self.learning_rate * step_gradient
            self.weights = unflatten_weights(weights_flat, self.layer_sizes)
            if i % 500 == 0:
                print(f"Iteration {i:>10,}: loss = {loss:>6.3f}")

    def forward(self, X):
        "Outcome of the model as 'soft predictions'."
        for W in self.weights:
            Z = X @ W.T
            X = np.tanh(Z)
        return self.uncenter(Z)  # intentionally ignore final tanh

    def predict(self, X):
        "Hard class predictions"

        preds = self.forward(X)
        if preds.shape[1] == 1:
            return np.squeeze(preds > 0.5, 1).astype(int)
        else:
            return np.argmax(preds, axis=1)


class TorchNeuralNetClassifier(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes,
        X=None,
        y=None,
        max_iter=10_000,
        learning_rate=0.0001,
        init_scale=1,
        batch_size=1,
        weight_decay=0,
        device=None,
    ):
        # This isn't really the typical way you'd lay out a pytorch module;
        # usually, you separate building the model and training it more.
        # This layout is like what we did before, though, and it'll do.
        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

        if X is not None and y is not None:
            self.fit(X, y)

    def cast(self, ary):
        # pytorch defaults everything to float32, unlike numpy which defaults to float64.
        # it's easier to keep everything the same,
        # and most ML uses don't really need the added precision...
        # you could use torch.set_default_dtype,
        # or pass dtype parameters everywhere you create a tensor, if you do want float64
        return torch.as_tensor(ary, dtype=torch.get_default_dtype(), device=self.device)

    def center(self, y):
        # change a [0, 1] variable to a [-1, 1] one
        return 2 * self.cast(y) - 1

    def uncenter(self, y):
        # convert from [-1, 1]-type labels to [0, 1]
        return (self.cast(y) + 1) / 2

    def loss_function(self):
        return nn.MSELoss()

    def build(self, in_dim, out_dim):
        layer_sizes = [in_dim] + list(self.hidden_layer_sizes) + [out_dim]

        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_size, out_size, device=self.device)
            nn.init.normal_(lin.weight, mean=0, std=self.init_scale)
            layers.append(lin)
            layers.append(nn.Tanh())

        layers.pop(-1)  # drop the final tanh

        self.layers = nn.Sequential(*layers)

    def make_optimizer(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def forward(self, x):
        return self.layers.forward(self.cast(x))

    def fit(self, X, y):
        X = self.cast(X)

        y = self.cast(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        y = self.center(y)

        self.build(X.shape[1], y.shape[1])

        loss_fn = self.loss_function()
        self.optimizer = self.make_optimizer()

        for i in range(self.max_iter):
            # Not doing anything fancy here like early stopping, etc.
            self.optimizer.zero_grad()

            inds = torch.as_tensor(
                np.random.choice(X.shape[0], size=self.batch_size, replace=False)
            )
            yhat = self(X[inds])
            loss = loss_fn(yhat, y[inds])

            loss.backward()
            self.optimizer.step()

            if i % 500 == 0:
                print(f"Iteration {i:>10,}: loss = {loss:>6.3f}")

    def predict(self, X):
        # hard class predictions (for soft ones, just call the model like the Z line below)
        with torch.no_grad():
            Z = self(X)

            preds = self.uncenter(Z).cpu().numpy()

            if preds.shape[1] == 1:
                return (np.squeeze(preds, 1) > 0.5).astype(int)
            else:
                return np.argmax(preds, 1)


class Convnet(TorchNeuralNetClassifier):
    def __init__(self, **kwargs):
        # default the hidden_layer_sizes thing to None
        kwargs.setdefault("hidden_layer_sizes", None)
        super().__init__(**kwargs)


    def forward(self, x):
        # hardcoding the shape of the data here, kind of obnoxiously.
        # normally we wouldn't have flattened the data in the first place,
        # just passed it in with shape e.g. (batch_size, channels, height, width)
        assert len(x.shape) == 2 and x.shape[1] == 784
        unflattened = x.reshape((x.shape[0], 1, 28, 28))
        return super().forward(unflattened)

    def build(self, in_dim, out_dim):
        # assign self.layers to an nn.Sequential with some Conv2d (and other) layers in it
        raise NotImplementedError()

