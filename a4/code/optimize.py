import numpy as np
from numpy.linalg import norm
from scipy.optimize import approx_fprime


def check_gradient(fun_and_grad, w, f, g, *args):
    """Similar to scipy.optimize.check_grad"""
    assert len(g.shape) == 1
    est_grad = approx_fprime(
        w, (lambda w: fun_and_grad(w, *args)[0]), epsilon=1.4901161193847656e-08
    )

    est_mask = np.isfinite(est_grad)
    g_mask = np.isfinite(g)
    if np.any(g_mask != est_mask):
        raise ValueError(
            "User and numerical derivatives have different nan shapes: "
            f"{(~est_mask).sum()} estimate aren't finite, {(~g_mask).sum()} in implemented"
        )

    vs = np.random.default_rng().normal(0, 1, size=(10, g_mask.sum()))
    vs /= np.linalg.norm(vs, axis=1, keepdims=True)

    est_vs = vs @ est_grad[est_mask]
    imp_vs = vs @ g[g_mask]

    if not np.allclose(est_vs, imp_vs, atol=0.01, rtol=0.01):
        index = np.nanargmax(np.abs(est_grad - g))
        lo = max(index - 2, 0)
        hi = min(index + 3, g.shape[0])
        raise ValueError(
            "User and numerical derivatives differ. "
            f"Showing positions {lo} to {hi - 1} (max diff at {index}):\n"
            f"est {est_grad[lo:hi]}\n"
            f"imp {g[lo:hi]}"
        )


def find_min(
    fun_and_grad,
    w,
    *args,
    max_evals=1000,
    verbosity=0,
    opt_tol=0.001,
    gamma=0.0001,
    check_grad=False,
):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha.

    If check_grad is passed, checks the gradient every check_grad steps (1 if True);
    will be very slow!
    """
    if check_grad is True:
        check_grad = 1

    f, g = fun_and_grad(w, *args)
    g = np.atleast_1d(g)

    fun_evals = 1
    if check_grad:
        if fun_evals % check_grad == 0:
            check_gradient(fun_and_grad, w, f, g, *args)
    alpha = 1.0

    while True:
        gg = g.T.dot(g)
        while True:
            w_new = w - alpha * g
            f_new, g_new = fun_and_grad(w_new, *args)
            fun_evals += 1
            if check_grad:
                if fun_evals % check_grad == 0:
                    check_gradient(fun_and_grad, w_new, f_new, g_new, *args)
            if f_new <= f - gamma * alpha * gg:
                break
            if verbosity > 1:
                print(f"f_new: {f_new:.3f} - f: {f:.3f} - Backtracking...")
            if np.isfinite(f_new):
                alpha = alpha**2 * gg / (2.0 * (f_new - f + alpha * gg))
            else:
                alpha /= 2

        if verbosity > 0:
            print(f"{fun_evals:,} - loss: {f_new:.3f}")
        y = g_new - g
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)
        if not np.isnan(alpha):
            if alpha < 1e-10 or alpha > 10000000000.0:
                alpha = 1.0
            if verbosity > 1:
                print(f"alpha: {alpha:.3f}")
            w = w_new
            f = f_new
            g = g_new
            opt_cond = norm(g, float("inf"))
            if opt_cond < opt_tol:
                if verbosity:
                    print(
                        f"Problem solved up to optimality tolerance {opt_tol} with {fun_evals:,} evals"
                    )
                break
            if fun_evals >= max_evals:
                if verbosity:
                    print(
                        f"Reached maximum number of function evaluations {max_evals:,}\n(opt condition {opt_cond} still > tolerance {opt_tol})"
                    )
                break

    return (w, f)


def find_min_L1(
    fun_and_grad,
    w,
    L1_lambda,
    max_evals,
    *args,
    verbosity=0,
    opt_tol=0.01,
    gamma=0.0001,
    check_grad=False,
):
    """
    Uses the L1 proximal gradient descent to optimize the objective function

    The line search algorithm divides the step size by 2 until
    it find the step size that results in a decrease of the L1 regularized
    objective function
    """
    if check_grad is True:
        check_grad = 1
    f, g = fun_and_grad(w, *args)
    fun_evals = 1
    if check_grad:
        if fun_evals % check_grad == 0:
            check_gradient(fun_and_grad, w, f, g, *args)
    alpha = 1.0
    proxL1 = lambda w, alpha: np.sign(w) * np.maximum(abs(w) - L1_lambda * alpha, 0)
    L1Term = lambda w: L1_lambda * np.sum(np.abs(w))
    while 1:
        gtd = None
        while True:
            w_new = w - alpha * g
            w_new = proxL1(w_new, alpha)
            if gtd is None:
                gtd = g.T.dot(w_new - w)
            f_new, g_new = fun_and_grad(w_new, *args)
            fun_evals += 1
            if check_grad:
                if fun_evals % check_grad == 0:
                    check_gradient(fun_and_grad, w_new, f_new, g_new, *args)
            if f_new + L1Term(w_new) <= f + L1Term(w) + gamma * alpha * gtd:
                break
            if verbosity > 1:
                print(f"f_new: {f_new:.3f} - f: {f:.3f} - Backtracking...")
            alpha /= 2.0

        if verbosity > 0:
            print(f"{fun_evals:,} - alpha {alpha:.3f} - loss: {f_new:.3f}")
        y = g_new - g
        alpha = -alpha * np.dot(y.T, g) / np.dot(y.T, y)
        if not np.isnan(alpha):
            if alpha < 1e-10 or alpha > 10000000000.0:
                alpha = 1.0
            w = w_new
            f = f_new
            g = g_new
            opt_cond = norm(w - proxL1(w - g, 1.0), float("inf"))
            if opt_cond < opt_tol:
                if verbosity:
                    print(
                        f"Problem solved up to optimality tolerance {opt_tol:.3f} with {fun_evals:,} evals"
                    )
                break
            if fun_evals >= max_evals:
                if verbosity:
                    print(
                        f"Reached maximum number of function evaluations {max_evals:,}\n(opt condition {opt_cond} still > tolerance {opt_tol})"
                    )
                break

    return (w, f)
