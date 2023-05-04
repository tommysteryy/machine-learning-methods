from functools import lru_cache

import numpy as np
from scipy import linalg, special

from optimize import find_min


class MultivariateT:
    # It would be nicer (and probably optimize faster) to instead parameterize
    # with Sigma = L @ L.T for triangular L, using self.use_full_flat=False below.
    # Unfortunately, I can't quite get the derivative to be right; I must be doing something wrong,
    # probably in C_to_Sigma_jacobian, but it's late and I'm not sure exactly what.
    # +5 bonus points to the first group who fully explains the error
    # and posts a correct (similar) version on Piazza.
    # (This is the kind of problem that autodiff totally dodges...)

    def __init__(self, X=None, use_full_flat=True, **kwargs):
        self.use_full_flat = use_full_flat
        if X is not None:
            self.fit(X, **kwargs)

    def fit(self, X, eps=0.01, check_grad=False, verbosity=0):
        minimize = lambda *a, **kw: find_min(
            *a, **kw, check_grad=check_grad, verbosity=verbosity
        )[0]

        n, d = X.shape

        mu = np.zeros(d)

        if self.use_full_flat:
            sigma_flat = np.eye(d).flatten()  # NOTE: this doesn't enforce symmetry!
        else:
            sigma_flat = get_tril(np.eye(d))  # cholesky-like factor of the covariance
            # NOTE: we don't enforce that diagonal elements of L are positive!
            #       this is a little different than a cholesky, be careful
            #       (but they can't be nearly-zero, or else Sigma would be singular)

        dof = 3 * np.ones((1,))

        # We'll do a coordinate descent style optimization:
        mu_old = np.ones(d)
        while np.linalg.norm(mu - mu_old, ord=np.inf) > eps:
            mu_old = mu
            mu = minimize(
                lambda my_mu: self.NLL_and_grad(X, my_mu, sigma_flat, dof, 0), mu
            )
            sigma_flat = minimize(
                lambda my_sigma_flat: self.NLL_and_grad(X, mu, my_sigma_flat, dof, 1),
                sigma_flat,
            )
            dof = minimize(
                lambda my_dof: self.NLL_and_grad(X, mu, sigma_flat, my_dof, 2), dof
            )

        self.mu = mu

        self.sigma_flat = sigma_flat
        if self.use_full_flat:
            self.Sigma = sigma_flat.reshape((d, d))
            self.Sigma = (self.Sigma + self.Sigma.T) / 2
        else:
            L = un_tril(sigma_flat, d)
            self.Sigma = L @ L.T

        self.dof = dof.item()

    def log_prob(self, X):
        return -self.NLL_and_grad(X, self.mu, self.sigma_flat, self.dof, deriv_wrt=None)

    def NLL_and_grad(self, X, mu, sigma_flat, dof, deriv_wrt=None):
        n, d = X.shape

        # check parameters are valid
        dof = np.atleast_1d(dof)  # ensure it's an array for a second...

        try:
            if self.use_full_flat:
                Sigma = sigma_flat.reshape((d, d))
                Sigma = (Sigma + Sigma.T) / 2
                L = linalg.cholesky(Sigma, lower=True)
            else:
                L = un_tril(sigma_flat, d)
                if np.abs(np.diagonal(L)).min() < 1e-8:
                    raise linalg.LinAlgError("almost-singular L")

            if np.any(dof <= 0):
                raise ValueError(f"non-positive dof {dof!r}")

        except (linalg.LinAlgError, ValueError):
            if deriv_wrt is None:
                return np.inf
            elif deriv_wrt == 0:
                return np.inf, np.full_like(mu, np.nan)
            elif deriv_wrt == 1:
                return np.inf, np.full_like(sigma_flat, np.nan)
            elif deriv_wrt == 2:
                return np.inf, np.full_like(dof, np.nan)
            else:
                raise ValueError(f"bad deriv_wrt {deriv_wrt!r}")

        dof = dof.item()  # convert to standard python float

        centered = X - mu[np.newaxis, :]

        # efficient linear algebra with cholesky factors:
        #   since Sigma = L L^T,
        #   inv(Sigma) = inv(L)^T inv(L)
        #   so  x @ inv(Sigma) @ x = || inv(L) @ x ||^2
        # solve_triangular is way faster than regular solve, once you have L!
        solves = linalg.solve_triangular(L, centered.T, lower=True)  # shape (d, n)
        qforms = (solves**2).sum(axis=0)  # shape (n,)

        # log det(Sigma) = log det(L L^T) = 2 log |det(L)| = 2 sum(log(abs(diag(L))))
        log_const = (
            special.gammaln((dof + d) / 2)
            - special.gammaln(dof / 2)
            - (d / 2) * np.log(dof * np.pi)
            - np.sum(np.log(np.abs(np.diagonal(L))))
        )

        nlls = ((d + dof) / 2) * np.log1p(qforms / dof) - log_const  # shape (n,)

        if deriv_wrt is None:
            return nlls

        elif deriv_wrt == 0:
            # mu derivative
            mults = -(dof + d) / (dof + qforms)
            Sigmainv_cent = linalg.solve_triangular(
                L, solves, lower=True, trans="T"
            )  # shape (d, t)
            g = np.einsum("i,ji->j", mults, Sigmainv_cent)

            return nlls.sum(), g

        elif deriv_wrt == 1:
            # Sigma derivative
            mults = -0.5 * (dof + d) / (dof + qforms)
            Sigmainv_cent = linalg.solve_triangular(
                L, solves, lower=True, trans="T"
            )  # shape (d, n)
            g_square = np.einsum("i,ji,ki->jk", mults, Sigmainv_cent, Sigmainv_cent)

            # need the actual inverse here, unfortunately
            #   inv(Sigma) = inv(L)^T inv(L)
            inv_L = linalg.solve_triangular(L, np.eye(d), lower=True)
            g_square += (n / 2) * (inv_L.T @ inv_L)

            # g_square should be symmetric already, but this might numerically help a bit
            g_square = (g_square + g_square.T) / 2

            if self.use_full_flat:
                g = g_square.flatten()
            else:
                # g_square is the derivative wrt Sigma
                # to get derivative wrt L, use the chain rule:
                #   J_{loss_with_Sigma o L_to_Sigma}(L) = J_{loss_with_Sigma}(L_to_Sigma(L))  J_{L_to_Sigma}(L)
                # here J_{loss_with_Sigma} is shape 1 by d^2 (we're using Sigma.flat)
                # and J_{L_to_Sigma} is d^2 by d(d+1)//2
                # unfortunately, J_{L_to_Sigma}(L) is kind of unpleasant; see below
                g = g_square.ravel(order="F") @ C_to_Sigma_jacobian(L)

            return nlls.sum(), g

        elif deriv_wrt == 2:
            # dof derivative
            g = (
                np.log1p(qforms / dof).sum() / 2
                - ((dof + d) / (2 * dof * (dof + qforms)) * qforms).sum()
                - (n / 2) * special.polygamma(0, (dof + d) / 2)
                + (n / 2) * special.polygamma(0, dof / 2)
                + n * (d / (2 * dof))
            )
            return nlls.sum(), np.full((1,), g)

        else:
            raise ValueError("bad deriv_wrt {deriv_wrt!r}")


# All of these helpers below are for the triangular version of Sigma,
# which doesn't quite work. Probably one of them is wrong!


def get_tril(a):
    return a[np.tril_indices_from(a)]


def un_tril(tril, d):
    square = np.zeros((d, d), dtype=tril.dtype)
    square[np.tril_indices(d)] = tril
    return square


def C_to_Sigma_jacobian(C):
    # https://math.stackexchange.com/q/2158399
    #  except we want   d vec(C C^T) / d vech(C), so we lose the L_k on the left
    # could probably implement more efficiently...
    k, _ = C.shape  # assumed square

    eye = np.eye(k)
    K_k = commutation_mat(k)
    Dtri_k = triangular_duplication_mat(k)

    # print(f"{k=} {L_k.shape=} {K_k.shape=} {D_k.shape=} {Dtri_k.shape=} {C.shape=}")
    return (np.kron(C, eye) + np.kron(eye, C) @ K_k) @ Dtri_k


# these are all implemented slowly, but whatever, they should only run once each
@lru_cache(10)
def triangular_duplication_mat(n):
    L_n = elimination_mat(n)
    D_n = duplication_mat(n)
    return np.diagonal(D_n @ L_n)[:, np.newaxis] * D_n


@lru_cache(10)
def duplication_mat(n):
    out = np.zeros((n * n, n * (n + 1) // 2))
    for j in range(n):
        for i in range(j, n):
            u = np.zeros(n * (n + 1) // 2)
            u[j * n + i - ((j + 1) * j) // 2] = 1

            T = np.zeros((n, n), order="F")
            T[i, j] = 1
            T[j, i] = 1

            out += np.outer(T.ravel(order="F"), u)
    return out


@lru_cache(10)
def elimination_mat(n):
    out = np.zeros((n * (n + 1) // 2, n * n))
    for j in range(n):
        for i in range(j, n):
            u_ij = np.zeros((n * (n + 1)) // 2)
            u_ij[j * n + i - ((j + 1) * j) // 2] = 1

            E_ij = np.zeros((n, n), order="F")
            E_ij[i, j] = 1

            out += np.outer(u_ij, E_ij.ravel(order="F"))
    return out


@lru_cache(10)
def commutation_mat(n):
    w = np.arange(n * n).reshape((n, n), order="F").T.ravel(order="F")
    return np.eye(n * n)[w, :]
